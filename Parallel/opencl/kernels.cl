//Create sampler for image2d_t that doesnt interpolate points, and sets out of bound pixels to 0
__constant sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;

inline int square(int x) {
	return x * x;
}

inline float euclidean_distance(int x2, int x1, int y2, int y1) {
	return sqrt((float)(square(x2 - x1) + square(y2 - y1)));
}

inline bool is_in_bounds(int x, int y, int width, int height, int bSize) {
	return y >= 0 && y < height - bSize && x >= 0 && x < width - bSize;
}

inline int absolute_difference(int a, int b) {
	return a < b ? b - a : a - b;
}

inline int mean_absolute_diff(image2d_t curr, image2d_t ref, int2 currPoint, int2 refPoint, int bSize) {
	return sum_absolute_diff(curr, ref, currPoint, refPoint, bSize) / square(bSize);
}

int matrix_sum(image2d_t img, int2 p, int bSize) {
	int sum = 0;

	for (int i = 0; i < bSize; i++)
	{
		for (int j = 0; j < bSize; j++)
		{
			//Force value to be absolute to force the equality|x+y|<=|x|+|y| because xy=|x||y|=|xy|
			sum += abs(read_imageui(img, sampler, (int2)(p.x + i, p.y + j)).x);
		}
	}

	return sum;
}

int sum_absolute_diff(image2d_t curr, image2d_t ref, int2 currPoint, int2 refPoint, int bSize) {
	int curr_sum = matrix_sum(curr, currPoint, bSize);
	int ref_sum = matrix_sum(ref, refPoint, bSize);
	return absolute_difference(curr_sum, ref_sum);
}

int3 closest_inbound_neighbour(image2d_t prev, int2 currPoint, const int sWindow, int width, int height, int blockSize) {
	for (int row = -sWindow; row < sWindow; row += sWindow) {
		for (int col = -sWindow; col < sWindow; col += sWindow) {
			int2 refPoint = { currPoint.x + row, currPoint.y + col };
			//Check if the block is within the bounds to avoid incorrect values
			if (is_in_bounds(refPoint.x, refPoint.y, width, height, blockSize)) {
				return (int3)(row, col, matrix_sum(prev, refPoint, blockSize));
			}
		}
	}
}

__kernel void full_exhastive(
	__read_only image2d_t prev,
	__read_only image2d_t curr,
	const uint blockSize,
	uint width,
	uint height,
	__global int2 * motionVectors
)
{
	//Get position within work group and reference block in current frame
	const int x = get_global_id(0), y = get_global_id(1);
	const int2 currPoint = { x * blockSize, y * blockSize };

	//Get number of blocks spanning the x-axis for buffer indexing
	const int wB = get_global_size(0);
	int idx = x + y * wB;

	//Get Sum of current block 
	int current_err = matrix_sum(curr, currPoint, blockSize);

	//Get closest block that's in bounds 
	const int sWindow = blockSize;
	int3 closest = closest_inbound_neighbour(prev, currPoint, sWindow, width, height, blockSize);
	//int ref_err = closest.z;

	float distanceToBlock = FLT_MAX;
	int bestErr = INT_MAX, err;

	//Loop over all possible blocks within each macroblock
	//Start at the closest neighbour that is inbound
	for (int row = closest.x; row < sWindow; row++) {
		for (int col = closest.y; col < sWindow; col++) {
			int2 refPoint = { currPoint.x + row, currPoint.y + col };

			//Check if the block is within the bounds to avoid incorrect values
			if (is_in_bounds(refPoint.x, refPoint.y, width, height, blockSize)) {

				int ref_err = matrix_sum(prev, refPoint, blockSize);
				err = absolute_difference(current_err, ref_err);

				//Weight results to preffer closer macroblocks
				float newDistance = euclidean_distance(refPoint.x, currPoint.x, refPoint.y, currPoint.y);

				if (err < bestErr) {
					bestErr = err;
					distanceToBlock = newDistance;
					motionVectors[idx] = refPoint;
				}
				else if (err == bestErr && newDistance <= distanceToBlock) {
					distanceToBlock = newDistance;
					motionVectors[idx] = refPoint;
				}
			}
		}
	}
}

__kernel void naive_full_exhastive(
	__read_only image2d_t prev,
	__read_only image2d_t curr,
	const uint blockSize,
	uint width,
	uint height,
	__global int2 * motionVectors
)
{
	//Get position within work group and reference block in current frame
	const int x = get_global_id(0), y = get_global_id(1);
	const int2 currPoint = { x * blockSize, y * blockSize };

	//Get number of blocks spanning the x-axis for buffer indexing
	const int wB = get_global_size(0);
	int idx = x + y * wB;

	const int sWindow = blockSize;
	float distanceToBlock = FLT_MAX;
	float bestErr = FLT_MAX, err;

	//Loop over all possible blocks within each macroblock
	for (int row = -sWindow; row < sWindow; row++) {
		for (int col = -sWindow; col < sWindow; col++) {
			int2 refPoint = { currPoint.x + row, currPoint.y + col };

			//Check if the block is within the bounds to avoid incorrect values
			if (is_in_bounds(refPoint.x, refPoint.y, width, height, blockSize)) {
				err = sum_absolute_diff(curr, prev, currPoint, refPoint, blockSize);

				//Weight results to preffer closer macroblocks
				float newDistance = euclidean_distance(refPoint.x, currPoint.x, refPoint.y, currPoint.y);

				if (err < bestErr) {
					bestErr = err;
					distanceToBlock = newDistance;
					motionVectors[idx] = refPoint;
				}
				else if (err == bestErr && newDistance <= distanceToBlock) {
					distanceToBlock = newDistance;
					motionVectors[idx] = refPoint;
				}
			}
		}
	}
}

__kernel void motion_estimation_opt(
	__read_only image2d_t prev,
	__read_only image2d_t curr,
	uint blockSize,
	uint width,
	uint height,
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
				}
				else if (SAD == lowestSAD && newDistance < distanceToBlock) {
					distanceToBlock = newDistance;
					motionVectors[mvPos] = searchPos;
				}
			}
		}

	}
}