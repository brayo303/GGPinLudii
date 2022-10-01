package Block;

import ai.djl.ndarray.types.Shape;
import ai.djl.nn.Activation;
import ai.djl.nn.Blocks;
import ai.djl.nn.SequentialBlock;
import ai.djl.nn.convolutional.Conv3d;
import ai.djl.nn.core.Linear;
import ai.djl.nn.norm.BatchNorm;
import ai.djl.nn.norm.Dropout;

public class GGPBlock extends AbstractBlockGenerator {
    public SequentialBlock setNN(int z, int y, int x, int outSize, int StartFilterNum, Float dropout){
        System.out.println("convggp");
        SequentialBlock block = new SequentialBlock();
        int filter = StartFilterNum;
        for(int i = 0 ; i < 4; i++){
            block.add(
                    Conv3d.builder()
                            .setKernelShape(new Shape(1,3, 3))
                            .setFilters(filter)
                            .optPadding(new Shape(0,1,1))
                            .build()

            ).add(BatchNorm.builder().build()).add(Activation::relu);
        }
        block.add(Blocks.batchFlattenBlock());
        block.add(Linear.builder().setUnits(1024).build());
        block.add(BatchNorm.builder().build());
        block.add(Activation::relu);
        block.add(Blocks.batchFlattenBlock());
        block.add(Dropout.builder().optRate(dropout).build());
        block.add(Linear.builder().setUnits(512).build());
        block.add(BatchNorm.builder().build());
        block.add(Activation::relu);
        block.add(Blocks.batchFlattenBlock());
        block.add(Dropout.builder().optRate(dropout).build());
        block.add(outputBlock(outSize));







        return block;
    }

    public boolean isDropLast(){
        return true;
    }

}
