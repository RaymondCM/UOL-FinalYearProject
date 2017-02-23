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

	if(pos.x < width && pos.y < height) {
		int mvPos = x + y * blockCount;
		motionVectors[mvPos] = pos;

		//Use reference block at location x:x y:y
		int lowestSAD = 0;
		for (int i = 0; i < blockSize; i++) {
			for (int j = 0; j < blockSize; j++) {
				int2 refPos = { pos.x + i, pos.y + y };
				lowestSAD += abs_diff((int)read_imageui(prev, sampler, refPos).x, (int)read_imageui(curr, sampler, refPos).x);
			}
		}

		//Calculate SAD -VERTICAL SCAN
		int searchMacro = blockSize / 2;
		for (int row = -searchMacro; row <= searchMacro; row++)
		{
			for (int col = -searchMacro; col <= searchMacro; col++)
			{
				int SAD = 0;

				for (int i = 0; i < blockSize; i++) {
					for (int j = 0; j < blockSize; j++) {
						int2 refPos = { pos.x + i, pos.y + y };
						int2 searchPos = { refPos.x + row, refPos.y + col };
						SAD += abs_diff((int)read_imageui(prev, sampler, refPos).x, (int)read_imageui(curr, sampler, searchPos).x);
					}
				}
				
				if(SAD < lowestSAD) {
					lowestSAD = SAD;
					motionVectors[mvPos] = (int2)(pos.x+row, pos.y+col);
				}
			}
		}

	}
}

