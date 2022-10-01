package Block;

import ai.djl.ndarray.types.Shape;
import ai.djl.nn.Activation;
import ai.djl.nn.Blocks;
import ai.djl.nn.SequentialBlock;
import ai.djl.nn.convolutional.Conv3d;
import ai.djl.nn.core.Linear;
import ai.djl.nn.norm.Dropout;
import ai.djl.nn.pooling.Pool;


public class ConvolutionalBlock extends AbstractBlockGenerator {




    public SequentialBlock setNN(int z,int x,int y,int outSize,int StartFilterNum, Float dropout){
        System.out.println("conv");
        SequentialBlock block = new SequentialBlock();
        int filter = StartFilterNum;
        int convNumber = 2;
        int curIt=0;
        int inSize = x*y*z;
        int neuronSize= inSize>outSize?inSize:outSize;
        while(filter<1024){
            int shapex;
            int shapey;
            int shapez;
            if(x<3&&y<3&&z<3){
                break;
            }
            int poolX;
            int poolY;
            int poolZ;
            if(z>=2) {
                poolZ = 2;
            }else{
                poolZ= 1;
            }
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

            if(z>=3){
                if(z%2==0){
                    shapez=3;
                    z/=2;
                }else{
                    shapez=2;
                    z+=1;
                    z/=2;
                }
            }else{
                shapez=1;
            }


            block.add(
                    Conv3d.builder()
                            .setKernelShape(new Shape(shapez,shapey, shapex))
                            .setFilters(filter)
                            .optPadding(new Shape(1,1,1))
                            .build()

            ).add(Activation::relu);



            for(int i = 0 ; i <convNumber-1 ; i ++){
                block.add(
                        Conv3d.builder()
                                .setKernelShape(new Shape(3,3, 3))
                                .setFilters(filter)
                                .optPadding(new Shape(1,1,1))
                                .build()

                ).add(Activation::relu);
            }
            block.add(Pool.maxPool3dBlock(new Shape(poolZ, poolY, poolX), new Shape(poolZ, poolY, poolX)));


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
