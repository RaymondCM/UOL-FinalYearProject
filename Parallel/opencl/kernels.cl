#define width 800
#define height 600

//Create sampler for image2d_t that doesnt interpolate points, and sets out of bound pixels to 0
__constant sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;

__kernel void motion_estimation(
	__read_only image2d_t prev,
	__read_only image2d_t curr,
	uint blockSize,
	__global int2 * motionVectors
)
{
	//Get position of work group and calculate position
	const int x = get_global_id(0), y = get_global_id(1);
	const int2 pos = { x * blockSize, y * blockSize };
	const int blockCount = get_global_size(0);

	//For debugging check if within range (should always == True)
	if (pos.x < width && pos.y < height) {
		//Calculate position result motion vector should go in buffer
		int mvPos = x + y * blockCount;

		/*Optimisation One: 
			Assume ABS(SUM(A(i:i+blockSize,j:j+blockSize)) - SUM(A(i:i+blockSize,j:j+blockSize))) 
			Equals element wise Sum of Absolute differences (True in tests).
		Allows for single computation of reference block sum rather that re accessing each element
		and allows for lower number of computations for current frame block (SUMB)*/

		int SUMA = 0, SUMB = 0, i, j;
		for (i = 0; i <= blockSize; i++) {
			for (j = 0; j <= blockSize; j++) {
				int2 refPos = { pos.x + i, pos.y + j };
				SUMA += read_imageui(prev, sampler, refPos).x;
				SUMB += read_imageui(curr, sampler, refPos).x;
			}
		}

		//Check if current frame is perfect match (SAD == 0) and break early.
		if (abs(SUMA - SUMB) == 0) {
			motionVectors[mvPos] = pos;
			return;
		}

		//Search window defined the area to search around the reference block
		const int searchWindow = blockSize;

		/*Distance to block is used to select the nearest block to the reference
		block if two blocks have the same SAD error.*/
		float distanceToBlock = FLT_MAX;

		/*Pre allocate variables for loops. 
		Initialise lowestSAD to MAX to ensure a value is always written to mVecBuffer*/
		int row, col, lowestSAD = INT_MAX, SAD = 0;

		/*Optimisation Two: (check with university if novel optimisation of full search)
			Due to assumtion SUMA - SUMB | SUMB - SUMA = SAD the calculation of SAD through
			vertical scan can be represented as SUMB0,1 = SUMB0,0 - SUMB0,0(:,0) + SUMB0,0(:,blockSize) 
			and for changes in y SUMB1,0 = SUMB0,0 - SUMB0,0(0,:) + SUMB0,0(blockSize,:).
			only requires blockSize * 2 calculations per block rather than (blockSize * blockSize) ^ 2*/

		//Calculate initial SUMB of top left corner (starting position of below loop)
		//TODO: Lowest SAD will always = SUMB below so maybe save one loop iteration and start searchWindow loop from -blockSize + 1
		for (i = 0; i <= blockSize; i++) {
			for (j = 0; j <= blockSize; j++) {
				SUMB += read_imageui(curr, sampler, (int2)(pos.x - searchWindow + i, pos.y - searchWindow + j)).x;
			}
		}

		/*Loop over every possible block in search window, save motion vector if point is the lowest SAD
			due to optimisation two only the edges of each frame need to be recalculated each loop due to
			all other elements SUMB is comprised off being shared between previous searched block and current*/
		for (row = -searchWindow; row <= searchWindow; row++)
		{
			col = -searchWindow;

			for (j = 0; j <= blockSize; j++) {
				//TODO: Condense these two operations into one.
				SUMB -= read_imageui(curr, sampler, (int2)(pos.x + row, pos.y + col + j)).x;
				SUMB += read_imageui(curr, sampler, (int2)(pos.x + row + blockSize, pos.y + col + j)).x;
			}

			for (; col <= searchWindow; col++)
			{
				int2 searchPos = { pos.x + row, pos.y + col };
				SUMB = 0;

				for (i = 0; i <= blockSize; i++) {
						SUMB -= read_imageui(curr, sampler, (int2)(pos.x + row + i, pos.y + col)).x;
						SUMB += read_imageui(curr, sampler, (int2)(pos.x + row + i, pos.y + col + blockSize)).x;
				}

				//TODO: Replace abs(SUMA - SUMB) with SUMA < SUMB ? SUMB - SUMA : SUMA - SUMB
				SAD = abs(SUMA - SUMB);

				//Calculate distance between search block and reference block to determine the closest match if two SAD are equal
				float newDistance = sqrt((float)(((searchPos.x - pos.x) * (searchPos.x - pos.x)) + ((searchPos.y - pos.y) * (searchPos.y - pos.y))));

				//Write buffer with lowestSAD location
				if (SAD < lowestSAD) {
					lowestSAD = SAD;
					distanceToBlock = newDistance;
					motionVectors[mvPos] = searchPos;
				} else if(SAD == lowestSAD && newDistance < distanceToBlock) {
					distanceToBlock = newDistance;
					motionVectors[mvPos] = searchPos;
				}
			}
		}

	}
}

