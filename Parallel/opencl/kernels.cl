__kernel void add(__global const int *A, __global const int *B, __global int *C)
{
    int id = get_global_id(0);
    if (id == 0)
    { //perform this part only once i.e. for work item 0
	printf("work group size %d\n", get_local_size(0));
    }
    int loc_id = get_local_id(0);
    printf("global id = %d, local id = %d\n", id, loc_id); //do it for each work item
    C[id] = A[id] + B[id];
}