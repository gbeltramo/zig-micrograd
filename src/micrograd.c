#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <stdio.h>
#include "capi.h"


typedef struct {
    PyObject_HEAD
    mg_dag *dag;
} GraphObject;

typedef struct {
    PyObject_HEAD
    GraphObject *graph; /* strong ref: keeps the arena alive */
    mg_node *node;
} ValueObject;

static PyTypeObject GraphType;
static PyTypeObject ValueType;
static GraphObject *g_default = NULL;

static PyObject *Graph_new(PyTypeObject *type, PyObject *args, PyObject *kwds) {
    GraphObject *self = (GraphObject *)type->tp_alloc(type, 0);
    if (!self) return NULL;
    self->dag = mg_dag_new();
    if (!self->dag) {
        Py_DECREF(self);
        return PyErr_NoMemory();
    }
    return (PyObject *)self;
}

static void Graph_dealloc(GraphObject *self) {
    if (self->dag) mg_dag_free(self->dag);
    Py_TYPE(self)->tp_free((PyObject *)self);
}

static PyTypeObject GraphType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "zig_micrograd._core.Graph",
    .tp_basicsize = sizeof(GraphObject),
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_doc = "Computation graph (owns all nodes created in it).",
    .tp_new = Graph_new,
    .tp_dealloc = (destructor)Graph_dealloc,
};

/* borrowed reference, NULL + exception on failure */
static GraphObject *get_default_graph(void) {
    if (!g_default) {
        g_default = (GraphObject *)PyObject_CallObject((PyObject *)&GraphType, NULL);
    }
    return g_default;
}


static PyObject *wrap(GraphObject *g, mg_node *node) {
    ValueObject *v = (ValueObject *)ValueType.tp_alloc(&ValueType, 0);
    if (!v) return NULL;
    Py_INCREF(g);
    v->graph = g;
    v->node = node;
    return (PyObject *)v;
}

static PyObject *Value_new(PyTypeObject *type, PyObject *args, PyObject *kwds) {
    static char *kwlist[] = {"data", "graph", NULL};
    double data;
    PyObject *gobj = NULL;
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "d|O!", kwlist, &data,
                                     &GraphType, &gobj))
        return NULL;
    GraphObject *g = gobj ? (GraphObject *)gobj : get_default_graph();
    if (!g) return NULL;
    mg_node *n = mg_leaf(g->dag, data);
    if (!n) return PyErr_NoMemory();
    return wrap(g, n);
}

static void Value_dealloc(ValueObject *self) {
    Py_XDECREF(self->graph);
    Py_TYPE(self)->tp_free((PyObject *)self);
}

/* 0 = ok, 1 = NotImplemented, -1 = error set */
static int to_node(PyObject *o, GraphObject *g, mg_node **out) {
    if (PyObject_TypeCheck(o, &ValueType)) {
        ValueObject *v = (ValueObject *)o;
        if (v->graph != g) {
            PyErr_SetString(PyExc_ValueError,
                            "Values belong to different graphs");
            return -1;
        }
        *out = v->node;
        return 0;
    }
    if (PyFloat_Check(o) || PyLong_Check(o)) {
        double d = PyFloat_AsDouble(o);
        if (d == -1.0 && PyErr_Occurred()) return -1;
        mg_node *n = mg_leaf(g->dag, d);
        if (!n) {
            PyErr_NoMemory();
            return -1;
        }
        *out = n;
        return 0;
    }
    return 1;
}

typedef mg_node *(*binop_fn)(mg_dag *, mg_node *, mg_node *);

static PyObject *binary(PyObject *a, PyObject *b, binop_fn fn) {
    GraphObject *g;
    if (PyObject_TypeCheck(a, &ValueType))
        g = ((ValueObject *)a)->graph;
    else if (PyObject_TypeCheck(b, &ValueType))
        g = ((ValueObject *)b)->graph;
    else
        Py_RETURN_NOTIMPLEMENTED;

    mg_node *na, *nb;
    int r = to_node(a, g, &na);
    if (r < 0) return NULL;
    if (r > 0) Py_RETURN_NOTIMPLEMENTED;
    r = to_node(b, g, &nb);
    if (r < 0) return NULL;
    if (r > 0) Py_RETURN_NOTIMPLEMENTED;

    mg_node *out = fn(g->dag, na, nb);
    if (!out) return PyErr_NoMemory();
    return wrap(g, out);
}

static PyObject *Value_add(PyObject *a, PyObject *b) { return binary(a, b, mg_add); }
static PyObject *Value_sub(PyObject *a, PyObject *b) { return binary(a, b, mg_sub); }
static PyObject *Value_mul(PyObject *a, PyObject *b) { return binary(a, b, mg_mul); }
static PyObject *Value_div(PyObject *a, PyObject *b) { return binary(a, b, mg_div); }

static PyObject *Value_neg(PyObject *a) {
    ValueObject *v = (ValueObject *)a;
    mg_node *n = mg_neg(v->graph->dag, v->node);
    if (!n) return PyErr_NoMemory();
    return wrap(v->graph, n);
}

/* like micrograd: only Value ** (int|float) */
static PyObject *Value_pow(PyObject *a, PyObject *b, PyObject *mod) {
    if (mod != Py_None) Py_RETURN_NOTIMPLEMENTED;
    if (!PyObject_TypeCheck(a, &ValueType)) Py_RETURN_NOTIMPLEMENTED;
    if (!(PyFloat_Check(b) || PyLong_Check(b))) Py_RETURN_NOTIMPLEMENTED;
    double e = PyFloat_AsDouble(b);
    if (e == -1.0 && PyErr_Occurred()) return NULL;
    ValueObject *v = (ValueObject *)a;
    mg_node *n = mg_pow(v->graph->dag, v->node, e);
    if (!n) return PyErr_NoMemory();
    return wrap(v->graph, n);
}

static PyNumberMethods Value_as_number = {
    .nb_add = Value_add,
    .nb_subtract = Value_sub,
    .nb_multiply = Value_mul,
    .nb_true_divide = Value_div,
    .nb_negative = Value_neg,
    .nb_power = Value_pow,
};

static PyObject *Value_relu(ValueObject *self, PyObject *Py_UNUSED(ignored)) {
    mg_node *n = mg_relu(self->graph->dag, self->node);
    if (!n) return PyErr_NoMemory();
    return wrap(self->graph, n);
}

static PyObject *Value_backward(ValueObject *self, PyObject *Py_UNUSED(ignored)) {
    if (mg_backward(self->graph->dag, self->node) != 0) return PyErr_NoMemory();
    Py_RETURN_NONE;
}

static PyMethodDef Value_methods[] = {
    {"relu", (PyCFunction)Value_relu, METH_NOARGS, "ReLU activation."},
    {"backward", (PyCFunction)Value_backward, METH_NOARGS,
     "Backpropagate from this node."},
    {NULL, NULL, 0, NULL},
};

static PyObject *Value_get_data(ValueObject *self, void *closure) {
    return PyFloat_FromDouble(mg_data(self->node));
}
static PyObject *Value_get_grad(ValueObject *self, void *closure) {
    return PyFloat_FromDouble(mg_grad(self->node));
}

static PyGetSetDef Value_getset[] = {
    {"data", (getter)Value_get_data, NULL, "forward value", NULL},
    {"grad", (getter)Value_get_grad, NULL, "gradient", NULL},
    {NULL, NULL, NULL, NULL, NULL},
};

static PyObject *Value_repr(ValueObject *self) {
    char buf[128];
    snprintf(buf, sizeof buf, "Value(data=%.17g, grad=%.17g)",
             mg_data(self->node), mg_grad(self->node));
    return PyUnicode_FromString(buf);
}

static PyTypeObject ValueType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "zig_micrograd._core.Value",
    .tp_basicsize = sizeof(ValueObject),
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_doc = "Scalar autograd value backed by the Zig DAG.",
    .tp_new = Value_new,
    .tp_dealloc = (destructor)Value_dealloc,
    .tp_repr = (reprfunc)Value_repr,
    .tp_as_number = &Value_as_number,
    .tp_methods = Value_methods,
    .tp_getset = Value_getset,
};


static PyObject *mod_reset(PyObject *self, PyObject *Py_UNUSED(ignored)) {
    Py_CLEAR(g_default);
    Py_RETURN_NONE;
}

static PyObject *mod_default_graph(PyObject *self, PyObject *Py_UNUSED(ignored)) {
    GraphObject *g = get_default_graph();
    if (!g) return NULL;
    Py_INCREF(g);
    return (PyObject *)g;
}

static PyMethodDef module_methods[] = {
    {"reset", mod_reset, METH_NOARGS,
     "Start a fresh default graph (old nodes are freed once unreferenced)."},
    {"default_graph", mod_default_graph, METH_NOARGS,
     "Return the current default Graph."},
    {NULL, NULL, 0, NULL},
};

static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "zig_micrograd._core",
    "Zig-backed micrograd engine.",
    -1,
    module_methods,
};

PyMODINIT_FUNC PyInit__core(void) {
    if (PyType_Ready(&GraphType) < 0) return NULL;
    if (PyType_Ready(&ValueType) < 0) return NULL;

    PyObject *m = PyModule_Create(&moduledef);
    if (!m) return NULL;

    Py_INCREF(&GraphType);
    if (PyModule_AddObject(m, "Graph", (PyObject *)&GraphType) < 0) {
        Py_DECREF(&GraphType);
        Py_DECREF(m);
        return NULL;
    }
    Py_INCREF(&ValueType);
    if (PyModule_AddObject(m, "Value", (PyObject *)&ValueType) < 0) {
        Py_DECREF(&ValueType);
        Py_DECREF(m);
        return NULL;
    }
    return m;
}
