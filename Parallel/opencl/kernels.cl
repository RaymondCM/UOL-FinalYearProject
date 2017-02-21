__kernel void shift(
    read_only image2d_t src,
    float shift_x,
    float shift_y,
    __global uchar *dst,
    int dst_step, int dst_offset, int dst_rows, int dst_cols)
{
    int x = get_global_id(0);
    int y = get_global_id(1);

    if (x >= dst_cols)
        return;
    int dst_index = mad24(y, dst_step, mad24(x, (int)sizeof(dstT), dst_offset));
    __global dstT *dstf = (__global dstT *)(dst + dst_index);
    float2 coord = (float2)((float)x + 0.5f + shift_x, (float)y + 0.5f + shift_y);

    dstf[0] = (dstT)read_imagef(src, CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_LINEAR, coord).x;
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
