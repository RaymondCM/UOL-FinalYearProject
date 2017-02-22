#define width 800
#define height 600 

__kernel void shift(__read_only image2d_t src, uint blockSize, __read_write  image2d_t dst)
{
    int x = get_global_id(0) * blockSize;
    int y = get_global_id(1) * blockSize;

	if(x < width && y < height) {
		//write_imageui(dst, (int2)(x, y), (uint4)x);
		printf("(X: %i, Y: %i)\n", x, y);
	}
	else {
		printf("Outside of bounds (X: %i, Y: %i)\n", x, y);
	}

    //printf("global id = %d, local id = %d\n", x, get_local_id(0)); //do it for each work item
}

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
