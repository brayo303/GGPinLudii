package Block;

import ai.djl.ndarray.NDList;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.Activation;
import ai.djl.nn.Blocks;
import ai.djl.nn.ParallelBlock;
import ai.djl.nn.SequentialBlock;
import ai.djl.nn.convolutional.Conv3d;
import ai.djl.nn.core.Linear;
import ai.djl.nn.norm.Dropout;
import ai.djl.nn.pooling.Pool;

import java.util.Arrays;

public abstract class AbstractBlockGenerator {
    SequentialBlock policyHead;
    SequentialBlock valueHead;


    boolean isDropLast()
    {
        return false;
    }

    public  abstract SequentialBlock setNN(int z, int y, int x, int outSize, int StartFilterNum, Float dropout);
    public ParallelBlock outputBlock(int outputUnitSize) {
        policyHead = new SequentialBlock();
        policyHead.add(Linear.builder().setUnits(outputUnitSize).build());
        valueHead = new SequentialBlock();
        valueHead.add(Linear.builder().setUnits(1).build());
        valueHead.add(Activation::tanh);


        return new ParallelBlock(
                list -> {
                    NDList ndlist = new NDList();
                    for(int i = 0 ; i < list.size() ; i++){
                        ndlist.addAll(list.get(i));
                    }
                    return ndlist;
                }, Arrays.asList(policyHead,valueHead));
    }
}
