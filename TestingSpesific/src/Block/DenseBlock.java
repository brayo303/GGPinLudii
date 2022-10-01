package Block;

import ai.djl.nn.Activation;
import ai.djl.nn.Blocks;
import ai.djl.nn.SequentialBlock;
import ai.djl.nn.core.Linear;
import ai.djl.nn.norm.Dropout;


public class DenseBlock extends AbstractBlockGenerator {
    public SequentialBlock setNN(int z, int y, int x, int outSize, int depth, Float dropout){
        System.out.println("seq");
        SequentialBlock block = new SequentialBlock();
        block.add(Blocks.batchFlattenBlock());
        int neuronnumber = outSize*outSize;
        for(int i = 0 ; i < depth ; i++) {
            //System.out.println(neuronnumber);
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
