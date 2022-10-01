package Block;

import ai.djl.ndarray.types.Shape;
import ai.djl.nn.Activation;
import ai.djl.nn.Blocks;
import ai.djl.nn.SequentialBlock;
import ai.djl.nn.convolutional.Conv3d;
import ai.djl.nn.core.Linear;
import ai.djl.nn.norm.Dropout;
import ai.djl.nn.pooling.Pool;

public class ConvolutionalBlockC extends AbstractBlockGenerator {

    public SequentialBlock setNN(int z, int y, int x, int outSize, int StartFilterNum, Float dropout){
        System.out.println("convc");
        SequentialBlock block = new SequentialBlock();
        int filter = StartFilterNum;
        int inSize = x*y*z;
        int neuronSize= inSize>outSize?inSize:outSize;
        int curIt=0;
        int convNumber = 2;

        while(filter<1024){
            int shapex;
            int shapey;
            if(x<3&&y<3){
                break;
            }
            int poolX;
            int poolY;
            if(x>=2) {
                poolX = 2;
            }else{
                poolX= 1;
            }
            if(y>=2) {
                poolY = 2;
            }else{
                poolY= 1;
            }
            if(x>=3){

                if(x%2==0){
                    shapex=3;
                    x/=2;
                }else{
                    shapex=2;
                    x+=1;
                    x/=2;
                }
            }else{
                shapex=1;
            }
            if(y>=3){

                if(y%2==0){
                    //System.out.println("masuk");
                    shapey=3;
                    y/=2;
                }else{
                    shapey=2;
                    y+=1;
                    y/=2;
                }
            }else{
                shapey=1;
            }

            block.add(
                    Conv3d.builder()
                            .setKernelShape(new Shape(1,shapey, shapex))
                            .setFilters(filter)
                            .optPadding(new Shape(0,1,1))
                            .build()

            ).add(Activation::relu);



            for(int i = 0 ; i <convNumber-1 ; i ++){
                block.add(
                        Conv3d.builder()
                                .setKernelShape(new Shape(1,3, 3))
                                .setFilters(filter)
                                .optPadding(new Shape(0,1,1))
                                .build()

                ).add(Activation::relu);
            }

            block.add(Pool.maxPool3dBlock(new Shape(1, poolY, poolX), new Shape(1, poolY, poolX)));

            filter*=2;
            curIt++;
            if(curIt==convNumber){
                curIt=0;
                convNumber++;
            }
        }

        block.add(Blocks.batchFlattenBlock());

        for(int i = 0 ; i<2 ; i ++) {

            block.add(Linear
                            .builder()
                            .setUnits(neuronSize)
                            .build())
                    .add(Activation::relu);

            if (dropout != null) {
                block.add(Dropout
                        .builder()
                        .optRate(dropout)
                        .build());
            }
        }
        block.add(outputBlock(outSize));
        return block;
    }
}