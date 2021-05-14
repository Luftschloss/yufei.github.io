GPU相关记录：
1、warps or wavefronts：

Nvidia版本：
在NVidia GPU中，最基本的处理单元是所谓的SP(Streaming Processor)，而一颗NVidia GPU中，会有非常多的SP可以同时做计算；而数个SP会在附加一些其他单元，一起组成一个SM(Streaming Multiprocessor)。
几个SM则会在组成所谓的 TPC(Texture Processing Clusters)。在G80/G92的架构下，总共会有128个SP，以8个SP为一组，组成16个SM，再以两个SM 为一个TPC，共分成8个TPC来运作。而在新一代的GT200里，SP
则是增加到240个，还是以8个SP组成一个SM，但是改成以3个SM组成一个TPC，共10组TPC。
对应CUDA：
应该是没有TPC的那一层架构，而是只要根据GPU的SM、SP的数量和资源来调整就可以了。如果把CUDA的Grid-Block-Thread架构对应到实际的硬件上的话，会类似对应成GPU-Streaming Multiprocessor-Streaming Processor;
一整个Grid会直接丢给GPU来执行，而Block大致就是对应到SM，thread则大致对应到SP。当然，这个讲法并不是很精确，只是一个简单的比喻而已。

AMD 版本：
OPENCL架构
另外work-item对应硬件上的一个PE（processing element）,而一个work-group对应硬件上的一个CU（computing unit）。这种对应可以理解为，一个work-item不能被拆分到多个PE上处理；同样
一个work-group也不能拆分到多个CU上同时处理（忘了哪里看到的信息）。当映射到OpenCL硬件模型上时，每一个work-item运行在一个被称为处理基元（processing element）的抽象硬件单元上，其中每个处理基元
可以处理多个work-item(注：摘自《OpenCL异构计算》P87)。（如此而言，是不是说对于二维的globalx必须是localx的整数倍，globaly必须是localy的整数倍？那么如果我数据很大，work-item所能数量很多，如果
一个group中中work-item的数量不超过CU中PE的个数，那么group的数量就可能很多；如果我想让group数量小点，那work-item的数目就会很多，还能不能处理了呢？这里总是找不多一个权威的解释，还请高手指点！
针对group和item的问题）。
对应CUDA
组织多个workgroup,每个workgroup划分为多个thread.由于硬件的限制，比如cu中pe数量的限制，实际上workgroup中线程并不是同时执行的，而是有一个调度单位，同一个workgroup中的线程，按照调度单位分组，
然后一组一组调度硬件上去执行。这个调度单位在nvidia的硬件上称作warp,在AMD的硬件上称作wavefront，或者简称为wave

所以理解上可以简单总结如下：

SP:最基本的处理单元。GPU进行并行计算，也就是很多个sp同时做处理。现在SP的术语已经有点弱化了，而是直接使用thread来代替。一个SP对应一个thread

Warp：warp是SM调度和执行的基础概念，通常一个SM中的SP(thread)会分成几个warp(也就是SP在SM中是进行分组的，物理上进行的分组)，一般每一个WARP中有32个thread.这个WARP中的32个thread(sp)是一起工作的，执行相同的指令，如果没有这么多thread需要工作，那么这个WARP中的一些thread(sp)是不工作的（每一个线程都有自己的寄存器内存和local memory，一个warp中的线程是同时执行的，也就是当进行并行计算时，线程数尽量为32的倍数，如果线程数不上32的倍数的话；假如是1，则warp会生成一个掩码，当一个指令控制器对一个warp单位的线程发送指令时，32个线程中只有一个线程在真正执行，其他31个 进程会进入静默状态。）

SM:streaming multiprocessor(sm)：多个sp加上其他的一些资源组成一个sm, 其他资源也就是存储资源，共享内存，寄储器等。可见，一个SM中的所有SP是先分成warp的，是共享同一个memory和instruction unit（指令单元）。从硬件角度讲，一个GPU由多个SM组成（当然还有其他部分），一个SM包含有多个SP（以及还有寄存器资源，shared memory资源，L1cache，scheduler，SPU，LD/ST单元等等）

首先解释下Cuda中的名词：
Block: 相当于opencl 中的work-group
Thread：相当于opencl 中的work-item
SP:   相当于opencl 中的PE
SM:  相当于opencl 中的CU
warp: 相当于opencl 中的wavefront(简称wave).

2、Computer Shader
Dispatch(x,y,z):线程组大小x*y*z
numthreads(x,y,z):单个线程组中线程数量x*y*z，thread group内可以共享一些信息
--SV_GroupThreadID 表示该线程在该组内的位置
--SV_GroupID 表示整个组所分配的位置
--SV_DispatchThreadID 表示该线程在所有组的线程中的位置
--SV_GroupIndex 表示该线程在该组内的索引
SM4.5 允许numthreads最多768条线程
SM5.0 允许numthreads最多1024条线程

