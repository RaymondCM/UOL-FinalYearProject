#define width 800
#define height 600 

__kernel void motion_estimation(
	__read_only image2d_t prev, 
	__read_only image2d_t curr, 
	uint blockSize,
	__global int2 * motionVectors
)
{
	const int x = get_global_id(0), y = get_global_id(1);
	const int2 pos = {x * blockSize, y * blockSize };
	const int blockCount = get_global_size(0);

	if (x == 0 && y == 0) {
		//printf("Block Count: %i\n", blockCount);
	}

	if(pos.x < width && pos.y < height) {
		//write_imageui(dst, (int2)(x, y), (uint4)x);
		int mvPos = x + y * blockCount;
		//printf("(X: %i, Y: %i, MV: %i)\n", pos.x, pos.y, mvPos);
		motionVectors[mvPos] = pos;
	}
	else {
		//printf("Outside of bounds (X: %i, Y: %i)\n", pos.x, pos.y);
	}

    //printf("global id = %d, local id = %d\n", x, get_local_id(0)); //do it for each work item
}