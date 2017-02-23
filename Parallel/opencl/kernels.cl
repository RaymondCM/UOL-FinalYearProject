#define width 800
#define height 600 

__constant sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;

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
		
		int lowestSAD = INT_MAX;

		int baseSAD = 0;
		for (int i = 0; i < blockSize; i++) {
			for (int j = 0; j < blockSize; j++) {
				int2 ij = {pos.x + i, pos.y + j};
				baseSAD += (int)(read_imageui(prev, sampler, ij).x - read_imageui(curr, sampler, ij).x);
			}
		}

		//Calculate SAD
		int searchMacro = blockSize / 2;
		for (int row = -searchMacro; row <= searchMacro; row++)
		{
			for (int col = -searchMacro; col <= searchMacro; col++)
			{
				int2 sPos = {pos.x + row, pos.y + col};

				int SAD = 0;
				for (int i = 0; i < blockSize; i++) {
					for (int j = 0; j < blockSize; j++) {
						int2 ij = {sPos.x + i, sPos.y + j};
						SAD += (int)(read_imageui(prev, sampler, ij).x - read_imageui(curr, sampler, ij).x);
					}
				}
				
				if(SAD < lowestSAD) {
					lowestSAD = SAD;
					motionVectors[mvPos] = sPos;
				}
			}
		}

		if(baseSAD <= lowestSAD) {
			motionVectors[mvPos] = pos;
		}

		//printf("SAD: %i", sum);
	}
	else {
		//printf("Outside of bounds (X: %i, Y: %i)\n", pos.x, pos.y);
	}

    //printf("global id = %d, local id = %d\n", x, get_local_id(0)); //do it for each work item
}

