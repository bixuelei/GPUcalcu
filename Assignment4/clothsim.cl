#define DAMPING 0.02f

#define G_ACCEL (float4)(0.f, -9.81f, 0.f, 0.f)

#define WEIGHT_ORTHO	0.138f
#define WEIGHT_DIAG		0.097f
#define WEIGHT_ORTHO_2	0.069f
#define WEIGHT_DIAG_2	0.048f


#define ROOT_OF_2 1.4142135f
#define DOUBLE_ROOT_OF_2 2.8284271f

#define WIND_DIRECTION (float4)(0.f, 0.f, 1.f, 0.f)
#define WIND_FN(time) (5.f * sin(0.5f *time))


///////////////////////////////////////////////////////////////////////////////
// The integration kernel
// Input data:
// width and height - the dimensions of the particle grid
// d_pos - the most recent position of the cloth particle while...
// d_prevPos - ...contains the position from the previous iteration.
// elapsedTime      - contains the elapsed time since the previous invocation of the kernel,
// prevElapsedTime  - contains the previous time step.
// simulationTime   - contains the time elapsed since the start of the simulation (useful for wind)
// All time values are given in seconds.
//
// Output data:
// d_prevPos - Input data from d_pos must be copied to this array
// d_pos     - Updated positions
///////////////////////////////////////////////////////////////////////////////
__kernel void Integrate(unsigned int width,
	unsigned int height,
	__global float4* d_pos,
	__global float4* d_prevPos,
	float elapsedTime,
	float prevElapsedTime,
	float simulationTime) {

	// Make sure the work-item does not map outside the cloth
	if (get_global_id(0) >= width || get_global_id(1) >= height)
		return;

	unsigned int particleID = get_global_id(0) + get_global_id(1) * width;
	// Read the positions
	float4 x0 = d_pos[particleID];
	float4 xP = d_prevPos[particleID];
	float4 xT = x0;

	// This is just to keep every 8th particle of the first row attached to the bar
	if (particleID > width - 1 || (particleID & (7)) != 0) {

		// Compute the new one position using the Verlet position integration, taking into account gravity and wind
		float4 wind = WIND_DIRECTION * WIND_FN(simulationTime);
		float4 a0 = G_ACCEL + wind;
		float4 v0 = (prevElapsedTime == 0.f) ? 0.f : ((x0 - xP) / prevElapsedTime);
		xT = x0 + v0 * elapsedTime + 0.5f * a0 * elapsedTime * elapsedTime;

	}
	// Move the value from d_pos into d_prevPos and store the new one in d_pos
	d_prevPos[particleID] = x0;
	d_pos[particleID] = xT;
}



///////////////////////////////////////////////////////////////////////////////
// Input data:
// pos1 and pos2 - The positions of two particles
// restDistance  - the distance between the given particles at rest
//
// Return data:
// correction vector for particle 1
///////////////////////////////////////////////////////////////////////////////
float4 SatisfyConstraint(float4 pos1,
	float4 pos2,
	float restDistance) {
	float4 toNeighbor = pos2 - pos1;
	return (toNeighbor - normalize(toNeighbor) * restDistance);     //if distance is great than restdistance,pos1 will get close to pos2
}

struct TileDesc {
	bool isLeftEdge, isTopEdge, isRightEdge, isBottomEdge;
};

struct TileDesc getTileDesc(int x, int y, uint width, uint height, int radius) {
	struct TileDesc result;

	result.isLeftEdge = x < radius;
	result.isTopEdge = y < radius;
	result.isRightEdge = x >= width - radius;
	result.isBottomEdge = y >= height - radius;

	return result;
}

///////////////////////////////////////////////////////////////////////////////
// Input data:
// width and height - the dimensions of the particle grid
// restDistance     - the distance between two orthogonally neighboring particles at rest
// d_posIn          - the input positions
//
// Output data:
// d_posOut - new positions must be written here
///////////////////////////////////////////////////////////////////////////////

#define TILE_X 16
#define TILE_Y 16
#define HALOSIZE 2
#define INDEX(x,y) ((x) + ((y) * width))

__kernel __attribute__((reqd_work_group_size(TILE_X, TILE_Y, 1)))
__kernel void SatisfyConstraints(unsigned int width,
	unsigned int height,
	float restDistance,
	__global float4* d_posOut,
	__global float4 const* d_posIn) {

	__local float4 tile[TILE_Y + 2 * HALOSIZE][TILE_X + 2 * HALOSIZE];
	__local float4 tileOut[TILE_Y][TILE_X];

	const uint2 GID = { get_global_id(0), get_global_id(1) };
	const uint2 LID = { get_local_id(0), get_local_id(1) };
	const uint2 TID = LID + HALOSIZE;
	const struct TileDesc GTD1 = getTileDesc(GID.x, GID.y, width, height, 1);
	const struct TileDesc GTD2 = getTileDesc(GID.x, GID.y, width, height, 2);
	const struct TileDesc LTD = getTileDesc(LID.x, LID.y, TILE_X, TILE_Y, 2);

	// load inner
	tile[TID.y][TID.x] = d_posIn[INDEX(GID.x, GID.y)];
	tileOut[LID.y][LID.x] = 0.0f;

	// load halo
	if (LTD.isLeftEdge) {
		tile[LID.y + 2][LID.x + 0] = GTD2.isLeftEdge ? NAN : d_posIn[INDEX(GID.x - 2, GID.y)];
	}
	else if (LTD.isRightEdge) {
		tile[LID.y + 2][LID.x + 4] = GTD2.isRightEdge ? NAN : d_posIn[INDEX(GID.x + 2, GID.y)];
	}

	if (LTD.isTopEdge) {
		tile[LID.y + 0][LID.x + 2] = GTD2.isTopEdge ? NAN : d_posIn[INDEX(GID.x, GID.y - 2)];
	}
	else if (LTD.isBottomEdge) {
		tile[LID.y + 4][LID.x + 2] = GTD2.isBottomEdge ? NAN : d_posIn[INDEX(GID.x, GID.y + 2)];
	}

	if (LTD.isLeftEdge && LTD.isTopEdge) {
		tile[LID.y + 0][LID.x + 0] = (GTD2.isLeftEdge || GTD2.isTopEdge) ? NAN : d_posIn[INDEX(GID.x - 2, GID.y - 2)];
	}
	else if (LTD.isRightEdge && LTD.isTopEdge) {
		tile[LID.y + 0][LID.x + 4] = (GTD2.isRightEdge || GTD2.isTopEdge) ? NAN : d_posIn[INDEX(GID.x + 2, GID.y - 2)];
	}
	else if (LTD.isLeftEdge && LTD.isBottomEdge) {
		tile[LID.y + 4][LID.x + 0] = (GTD2.isLeftEdge || GTD2.isBottomEdge) ? NAN : d_posIn[INDEX(GID.x - 2, GID.y + 2)];
	}
	else if (LTD.isRightEdge && LTD.isBottomEdge) {
		tile[LID.y + 4][LID.x + 4] = (GTD2.isRightEdge || GTD2.isBottomEdge) ? NAN : d_posIn[INDEX(GID.x + 2, GID.y + 2)];
	}

	// sync threads
	barrier(CLK_LOCAL_MEM_FENCE);


	if (get_global_id(0) >= width || get_global_id(1) >= height)
		return;


	// This is just to keep every 8th particle of the first row attached to the bar
	if (INDEX(GID.x, GID.y) > width - 1 || (INDEX(GID.x, GID.y) & (7)) != 0) {
		// orthogonal constrains 1
		if (!GTD1.isRightEdge) {
			tileOut[LID.y][LID.x] += SatisfyConstraint(tile[TID.y][TID.x], tile[TID.y][TID.x + 1], restDistance) * WEIGHT_DIAG;
		}
		if (!GTD1.isLeftEdge) {
			tileOut[LID.y][LID.x] += SatisfyConstraint(tile[TID.y][TID.x], tile[TID.y][TID.x - 1], restDistance) * WEIGHT_DIAG;
		}
		// orthogonal constrains 2
		if (!GTD2.isRightEdge) {
			tileOut[LID.y][LID.x] += SatisfyConstraint(tile[TID.y][TID.x], tile[TID.y][TID.x + 2], restDistance * 2) * WEIGHT_DIAG_2;
		}
		if (!GTD2.isLeftEdge) {
			tileOut[LID.y][LID.x] += SatisfyConstraint(tile[TID.y][TID.x], tile[TID.y][TID.x - 2], restDistance * 2) * WEIGHT_DIAG_2;
		}

		// diagonal constrains 1
		if (!GTD1.isRightEdge && !GTD1.isBottomEdge) {
			tileOut[LID.y][LID.x] += SatisfyConstraint(tile[TID.y][TID.x], tile[TID.y + 1][TID.x + 1], ROOT_OF_2 * restDistance) * WEIGHT_ORTHO;
		}
		if (!GTD1.isLeftEdge && !GTD1.isBottomEdge) {
			tileOut[LID.y][LID.x] += SatisfyConstraint(tile[TID.y][TID.x], tile[TID.y + 1][TID.x - 1], ROOT_OF_2 * restDistance) * WEIGHT_ORTHO;
		}
		if (!GTD1.isRightEdge && !GTD1.isTopEdge) {
			tileOut[LID.y][LID.x] += SatisfyConstraint(tile[TID.y][TID.x], tile[TID.y - 1][TID.x + 1], ROOT_OF_2 * restDistance) * WEIGHT_ORTHO;
		}
		if (!GTD1.isLeftEdge && !GTD1.isTopEdge) {
			tileOut[LID.y][LID.x] += SatisfyConstraint(tile[TID.y][TID.x], tile[TID.y - 1][TID.x - 1], ROOT_OF_2 * restDistance) * WEIGHT_ORTHO;
		}
		// diagonal constrains 2
		if (!GTD2.isRightEdge && !GTD2.isBottomEdge) {
			tileOut[LID.y][LID.x] += SatisfyConstraint(tile[TID.y][TID.x], tile[TID.y + 2][TID.x + 2], DOUBLE_ROOT_OF_2 * restDistance) * WEIGHT_ORTHO_2;
		}
		if (!GTD2.isLeftEdge && !GTD2.isBottomEdge) {
			tileOut[LID.y][LID.x] += SatisfyConstraint(tile[TID.y][TID.x], tile[TID.y + 2][TID.x - 2], DOUBLE_ROOT_OF_2 * restDistance) * WEIGHT_ORTHO_2;
		}
		if (!GTD2.isRightEdge && !GTD2.isTopEdge) {
			tileOut[LID.y][LID.x] += SatisfyConstraint(tile[TID.y][TID.x], tile[TID.y - 2][TID.x + 2], DOUBLE_ROOT_OF_2 * restDistance) * WEIGHT_ORTHO_2;
		}
		if (!GTD2.isLeftEdge && !GTD2.isTopEdge) {
			tileOut[LID.y][LID.x] += SatisfyConstraint(tile[TID.y][TID.x], tile[TID.y - 2][TID.x - 2], DOUBLE_ROOT_OF_2 * restDistance) * WEIGHT_ORTHO_2;
		}
	}

	tileOut[LID.y][LID.x].w = 0.0f;
	const float len = sqrt(tileOut[LID.y][LID.x].x*tileOut[LID.y][LID.x].x+tileOut[LID.y][LID.x].y*tileOut[LID.y][LID.x].y+tileOut[LID.y][LID.x].z*tileOut[LID.y][LID.x].z);
	//const float len = length(tileOut[LID.y][LID.x]);

	// clamp max change rate to d/2
	if (len > restDistance / 2) {
		tileOut[LID.y][LID.x] *= restDistance / 2 / len;
	}

	// store updated pos
	d_posOut[INDEX(GID.x, GID.y)] = tile[TID.y][TID.x] + tileOut[LID.y][LID.x];
}


///////////////////////////////////////////////////////////////////////////////
// Input data:
// width and height - the dimensions of the particle grid
// d_pos            - the input positions
// spherePos        - The position of the sphere (xyz)
// sphereRad        - The radius of the sphere
//
// Output data:
// d_pos            - The updated positions
///////////////////////////////////////////////////////////////////////////////
__kernel void CheckCollisions(unsigned int width,
	unsigned int height,
	__global float4* d_pos,
	float4 spherePos,
	float sphereRad) {


	if (get_global_id(0) >= width || get_global_id(1) >= height)
		return;

	const unsigned int particleID = get_global_id(0) + get_global_id(1) * width;

	const float4 distanceVec = d_pos[particleID] - spherePos;
	const float squaredDistance = (distanceVec.x * distanceVec.x) + (distanceVec.y * distanceVec.y) + (distanceVec.z * distanceVec.z);

	// Find whether the particle is inside the sphere.
	if (squaredDistance < sphereRad * sphereRad) {
		// If so, push it outside.
		d_pos[particleID] += (sphereRad - sqrt(squaredDistance)) * normalize(distanceVec);
	}

}

///////////////////////////////////////////////////////////////////////////////
// There is no need to change this function!
///////////////////////////////////////////////////////////////////////////////
float4 CalcTriangleNormal(float4 p1, float4 p2, float4 p3) {
	float4 v1 = p2 - p1;
	float4 v2 = p3 - p1;

	return cross(v1, v2);
}

///////////////////////////////////////////////////////////////////////////////
// There is no need to change this kernel!
///////////////////////////////////////////////////////////////////////////////
__kernel void ComputeNormals(unsigned int width,
	unsigned int height,
	__global float4* d_pos,
	__global float4* d_normal) {

	int particleID = get_global_id(0) + get_global_id(1) * width;
	float4 normal = (float4)(0.0f, 0.0f, 0.0f, 0.0f);

	int minX, maxX, minY, maxY, cntX, cntY;
	minX = max((int)(0), (int)(get_global_id(0) - 1));
	maxX = min((int)(width - 1), (int)(get_global_id(0) + 1));
	minY = max((int)(0), (int)(get_global_id(1) - 1));
	maxY = min((int)(height - 1), (int)(get_global_id(1) + 1));

	for (cntX = minX; cntX < maxX; ++cntX) {
		for (cntY = minY; cntY < maxY; ++cntY) {
			normal += normalize(CalcTriangleNormal(
				d_pos[(cntX + 1) + width * (cntY)],
				d_pos[(cntX)+width * (cntY)],
				d_pos[(cntX)+width * (cntY + 1)]));
			normal += normalize(CalcTriangleNormal(
				d_pos[(cntX + 1) + width * (cntY + 1)],
				d_pos[(cntX + 1) + width * (cntY)],
				d_pos[(cntX)+width * (cntY + 1)]));
		}
	}
	d_normal[particleID] = normalize(normal);
}
