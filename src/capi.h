#ifndef MG_CAPI_H
#define MG_CAPI_H

#ifdef __cplusplus
extern "C" {
#endif

typedef struct mg_dag mg_dag;
typedef struct mg_node mg_node;

mg_dag *mg_dag_new(void);
void mg_dag_free(mg_dag *dag);

mg_node *mg_leaf(mg_dag *dag, double data);
mg_node *mg_add(mg_dag *dag, mg_node *a, mg_node *b);
mg_node *mg_sub(mg_dag *dag, mg_node *a, mg_node *b);
mg_node *mg_mul(mg_dag *dag, mg_node *a, mg_node *b);
mg_node *mg_div(mg_dag *dag, mg_node *a, mg_node *b);
mg_node *mg_neg(mg_dag *dag, mg_node *a);
mg_node *mg_pow(mg_dag *dag, mg_node *base, double exponent);
mg_node *mg_relu(mg_dag *dag, mg_node *a);

double mg_data(mg_node *n);
double mg_grad(mg_node *n);
int mg_backward(mg_dag *dag, mg_node *root); /* 0 ok, -1 OOM */

#ifdef __cplusplus
}
#endif
#endif
