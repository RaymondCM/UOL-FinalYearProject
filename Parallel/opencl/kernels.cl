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
		
		int referenceBlock = read_imageui(prev, sampler, pos).x;
		//int lowestSAD = INT_MAX;
		int baseSAD = 0;
		
		for (int i = 0; i < blockSize; i++) {
			for (int j = 0; j < blockSize; j++) {
				int2 ij = {pos.x + i, pos.y + j};
				baseSAD += abs_diff(referenceBlock, (int)read_imageui(curr, sampler, (int2)(pos.x + i, pos.y + j)).x);
			}
		}
		
		int lowestSAD = baseSAD;
		motionVectors[mvPos] = pos;
		

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
						SAD += abs_diff(referenceBlock, (int)read_imageui(curr, sampler, (int2)(sPos.x + i, sPos.y + j)).x);
					}
				}
				
				if(SAD < lowestSAD && SAD > 1000) {
					lowestSAD = SAD;
					motionVectors[mvPos] = sPos;
				}
			}
		}

		//printf("SAD: %i", sum);
	}
	else {
		//printf("Outside of bounds (X: %i, Y: %i)\n", pos.x, pos.y);
	}

    //printf("global id = %d, local id = %d\n", x, get_local_id(0)); //do it for each work item
}

