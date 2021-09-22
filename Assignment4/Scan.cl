


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
__kernel void ScanNaive(const __global uint* inArray, __global uint* outArray, uint N, uint offset)
{
	uint GID = get_global_id(0);
	uint GSize = get_global_size(0);
	uint LID = get_local_id(0);
	uint LGID = get_group_id(0);
	uint LSize = get_local_size(0);

	if(GID < offset) {
		outArray[GID] = inArray[GID];
	} else if(GID < N) {
		outArray[GID] = inArray[GID] + inArray[GID - offset];
	}
}




#define UNROLL
#define NUM_BANKS			32
#define NUM_BANKS_LOG		5
#define SIMD_GROUP_SIZE		32

// Bank conflicts
#define AVOID_BANK_CONFLICTS
#ifdef AVOID_BANK_CONFLICTS
	// simple offset spacing as in Assingment Sheet
	//#define OFFSET(A) ((A) + ((A)/NUM_BANKS))

	// see: http://hpc.isti.cnr.it/~gabriele/pdf/0520479724.pdf
	//#define RST(A) ((A) & (0xFFFFFFFF << NUM_BANKS))
	//#define ROT1(A) ((A) & ((1<<NUM_BANKS)-1))
	//#define ROT(A,NUM_BANKS) ((ROT1(x) >>NUM_BANKS) | (ROT1(x) << (NUM_BANKS-NUM_BANKS_LOG)))
	//#define LR(A, NUM_BANKS) (RST((x)*(1<<NUM_BANKS)) + ROT(x,m))
	//#define OFFSET(A) (LR((A),((A)/NUM_BANKS)))

	// introduced by official website
#define OFFSET(A)   ((A)+((A) >> NUM_BANKS + (A) >> (2 *  NUM_BANKS_LOG)))
#else
#define OFFSET(A) (A)
#endif

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
__kernel void Scan(__global uint* array, __global uint* higherLevelArray, __local uint* localBlock)
{
	// TO DO: Kernel implementation
	uint GID = get_global_id(0);
	uint GSize = get_global_size(0);
	uint LID = get_local_id(0);
	uint LGID = get_group_id(0);
	uint LSize = get_local_size(0);
	// 256 localworksize will process 512 elements, so the localsize should be double.

	uint LN = LSize * 2;

	// group wise loading

	localBlock[OFFSET(LID)] = array[LID + LGID * LN];
	localBlock[OFFSET(LID + LSize)] = array[LID + LGID * LN + LSize];
	barrier(CLK_LOCAL_MEM_FENCE);



	// up-sweep
#pragma unroll
	for (uint stride = 1; stride * 2 <= LN; stride *= 2) {
		if (LID < LN / (stride * 2)) {
			//localBlock[OFFSET(LN - 1 - LID * (stride * 2))] +=  localBlock[OFFSET(LN - 1 - LID * (stride * 2) - stride)];
			localBlock[OFFSET(stride * (2 * LID + 2) - 1)] += localBlock[OFFSET(stride * (2 * LID + 1) - 1)];

		}
		barrier(CLK_LOCAL_MEM_FENCE);

	}


	// down-sweep
	localBlock[OFFSET(LN - 1)] = 0;
	barrier(CLK_LOCAL_MEM_FENCE);
#pragma unroll
	for (uint stride = LN / 2; stride > 0; stride /= 2) {
		if (LID < LN / (stride * 2)) {
			uint tmp = localBlock[OFFSET(LN - 1 - LID * (stride * 2))];
			localBlock[OFFSET(LN - 1 - LID * (stride * 2))] += localBlock[OFFSET(LN - 1 - LID * (stride * 2) - stride)];
			localBlock[OFFSET(LN - 1 - LID * (stride * 2) - stride)] = tmp;
			//uint tmp = localBlock[OFFSET(stride * (2 * LID + 1) - 1)];
			//localBlock[OFFSET(stride * (2 * LID + 1) - 1)] += localBlock[OFFSET(stride * (2 * LID + 2) - 1)];
			//localBlock[OFFSET(stride * (2 * LID + 2) - 1)] += tmp;
		}
		barrier(CLK_LOCAL_MEM_FENCE);
	}

	//from exclusive to inclusive
	array[LID + LGID * LN] += localBlock[OFFSET(LID)];
	array[LID + LGID * LN + LSize] += localBlock[OFFSET(LID + LSize)];

	if (LID == LSize - 1) {
		// need to load from global memory because be do not have the correct value in local memory
		higherLevelArray[LGID] = array[LID + LGID * LN + LSize];

	}
}



//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
__kernel void ScanAdd(__global uint* higherLevelArray, __global uint* array, __local uint* localBlock)
{
	// TO DO: Kernel implementation (large arrays)
	uint LID = get_local_id(0);
	uint LGID = get_group_id(0);
	uint LSize = get_local_size(0);
	if (LGID >= 2) {
		array[LID + LGID * LSize] += higherLevelArray[LGID / 2 - 1];
	}
	;
}