package Block;

import ai.djl.ndarray.types.Shape;
import ai.djl.nn.Activation;
import ai.djl.nn.Blocks;
import ai.djl.nn.SequentialBlock;
import ai.djl.nn.convolutional.Conv3d;
import ai.djl.nn.core.Linear;
import ai.djl.nn.norm.Dropout;

public class NoMaxPoolBlock extends AbstractBlockGenerator{
    @Override
    public SequentialBlock setNN(int z, int x, int y, int outSize, int StartFilterNum, Float dropout) {
        int neuralnetworkdepth = (int)Math.sqrt(x*y);
        int filterNum = StartFilterNum;
        int limit = 1;
        SequentialBlock block = new SequentialBlock();
        for(int i = 0 ; i < neuralnetworkdepth ; i++){
            block.add(Conv3d.builder()
                    .setKernelShape(new Shape(1,3,3 ))
                    .setFilters(filterNum)
                    .optPadding(new Shape(0,1,1))
                    .build()).add(Activation::relu);
            if(i==limit){
                limit+=2;
                filterNum*=2;

            }
        }
        block.add(Blocks.batchFlattenBlock());

        int neuronnumber = x*y*z>outSize?x*y*z:outSize;

        for(int i = 0 ; i < 2 ; i++) {
            block.add(Linear.builder().setUnits(neuronnumber).build());
            block.add(Activation::relu);
            if(dropout!=null) {
                block.add(Dropout.builder().optRate(dropout).build());
            }
        }
        block.add(outputBlock(outSize));
        return block;

    }
}
