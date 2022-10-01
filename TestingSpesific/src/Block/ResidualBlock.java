package Block;

import ai.djl.ndarray.NDList;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.Activation;
import ai.djl.nn.Blocks;
import ai.djl.nn.ParallelBlock;
import ai.djl.nn.SequentialBlock;
import ai.djl.nn.convolutional.Conv2d;
import ai.djl.nn.core.Linear;
import ai.djl.nn.norm.BatchNorm;

import javax.sound.sampled.Line;
import java.util.Arrays;

public class ResidualBlock extends AbstractBlockGenerator{

    @Override
    public SequentialBlock setNN(int z, int y, int x, int outSize, int FilterNum, Float dropout) {
        SequentialBlock net = new SequentialBlock();
        net
                .add(
                        Conv2d.builder()
                                .setKernelShape(new Shape(3, 3))
                                .optStride(new Shape(1, 1))
                                .optPadding(new Shape(1, 1))
                                .setFilters(FilterNum)
                                .build())
                .add(BatchNorm.builder().build());
        net.add(Activation::relu);
        net.add(residualBlock(FilterNum));
        net.add(Blocks.batchFlattenBlock());
        net.add(Linear.builder().setUnits(4096).build());
        net.add(Activation::relu);
        net.add(Linear.builder().setUnits(4096).build());
        net.add(Activation::relu);
        net.add(outputBlock(outSize));
        return net;
    }


    public ParallelBlock residualBlock(int FilterNum) {
        SequentialBlock b1;
        SequentialBlock conv1x1;

        b1 = new SequentialBlock();
        b1.add(Conv2d.builder()
                        .setFilters(FilterNum)
                        .setKernelShape(new Shape(3, 3))
                        .optPadding(new Shape(1, 1))
                        .optStride(new Shape(1,1))
                        .build())
                .add(BatchNorm.builder().build())
                .add(Activation::relu)
                .add(Conv2d.builder()
                        .setFilters(FilterNum)
                        .setKernelShape(new Shape(3, 3))
                        .optPadding(new Shape(1, 1))
                        .optStride(new Shape(1,1))
                        .build())
                .add(BatchNorm.builder().build());
        conv1x1 = new SequentialBlock();
        conv1x1.add(Blocks.identityBlock());
        ParallelBlock block = new ParallelBlock(
                list -> {
                    NDList unit = list.get(0);
                    NDList parallel = list.get(1);
                    return new NDList(
                            unit.singletonOrThrow()
                                    .add(parallel.singletonOrThrow())
                                    .getNDArrayInternal()
                                    .relu());
                },
                Arrays.asList(b1, conv1x1));
        return block;

    }

    @Override
    boolean isDropLast() {
        return true;
    }
}
