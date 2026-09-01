#pragma once
#include <algorithm>
#include <atomic>
#include <cstddef>
#include <thread>
#include <vector>

/*
 * Runs an index range across the hardware threads.
 *
 * The whole of the engine's CPU parallelism, deliberately: the loader has one place where work
 * is worth spreading -- decoding a scene's images -- and one shape of it, so a work queue and a
 * join is the entire requirement. A general thread pool would exist to amortise thread creation
 * across many dispatches, which is not what this does: a scene load dispatches once.
 */
namespace Nexus::Utils {

	/*
	 * Calls body(i) exactly once for every i in [0, count), from an unspecified thread, and
	 * returns when all of them have finished.
	 *
	 * body must not touch CUDA or OpenGL, and must not mutate anything shared. Writing to
	 * element i of a container sized before the call is the intended shape: distinct elements of
	 * a std::vector are distinct objects, so those writes do not race. std::vector<bool> is the
	 * exception -- it packs its elements into bits, so neighbouring writes share a byte.
	 *
	 * body must not throw either: an exception escaping a std::thread terminates the process.
	 */
	template<typename TBody>
	void ParallelFor(size_t count, TBody body)
	{
		unsigned int threadCount = std::max(1u, std::thread::hardware_concurrency());
		threadCount = std::min(threadCount, static_cast<unsigned int>(count));

		// Also the count == 0 case, which the loop below would otherwise spawn nothing for.
		if (threadCount <= 1)
		{
			for (size_t i = 0; i < count; i++)
				body(i);
			return;
		}

		/*
		 * The work queue. Threads claim indices one at a time rather than taking a fixed slice
		 * each, because the per-index cost is not uniform -- decoding a 4K base colour map is
		 * orders of magnitude more work than a 256x256 roughness map, and a static split leaves
		 * whichever thread drew the large images still running long after the others finished.
		 *
		 * Relaxed ordering is all the counter itself needs: it decides only which thread takes
		 * which index, and nothing is published through it. The join below is what makes
		 * everything the threads wrote visible here.
		 */
		std::atomic<size_t> next{ 0 };

		std::vector<std::thread> workers;
		workers.reserve(threadCount);

		for (unsigned int t = 0; t < threadCount; t++)
		{
			workers.emplace_back([&body, &next, count]()
			{
				for (size_t i = next.fetch_add(1, std::memory_order_relaxed); i < count;
					i = next.fetch_add(1, std::memory_order_relaxed))
				{
					body(i);
				}
			});
		}

		for (std::thread& worker : workers)
			worker.join();
	}

}
