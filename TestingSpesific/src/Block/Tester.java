package Block;



import ai.djl.Model;


import ai.djl.engine.Engine;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.Block;
import ai.djl.nn.Parameter;

import ai.djl.training.DefaultTrainingConfig;


import ai.djl.training.Trainer;
import ai.djl.training.TrainingConfig;


import ai.djl.training.initializer.UniformInitializer;

import ai.djl.training.listener.*;

import java.util.ArrayList;


public class Tester {




    public static void main(String[] args){
        System.out.println(getParameterNumber());


    }

    public static void testLoadAll(){
        ArrayList<float[]> prevdiff[] = null;
        ArrayList<Integer> idx = new ArrayList<>();
        for(int i = 0 ; i < 100 ; i++) {
            NeuralNetworkManager curNet = new NeuralNetworkManager("D:/Eval/25it/", i, 2,
                    7,
                    7,
                    51,
                    0.01f,
                    10,
                    64,
                    null,
                    'a',
                    9,
                    'a',
                    new int[]{0, 1});

            if(prevdiff!=null) {
                if(curNet.debugTest(prevdiff, curNet.getLastWeight())){
                    idx.add(i);
                }
            }
            prevdiff=curNet.getLastWeight();
            curNet.closeModel();
        }
        System.out.println("aaaaaaaaaa");
        for(int i = 0 ; i  < idx.size() ; i++){
            System.out.print(idx.get(i)+" ");
        }
    }

    public static void testShape(){
        NDManager manager = Engine.getInstance().newBaseManager();
        NDArray X = manager.randomUniform(0f, 1.0f, new Shape( 1,1 ,2, 6, 6));
        Model model = Model.newInstance("model");
        AbstractBlockGenerator cblock = new ConvolutionalBlockC();
        Block block = cblock.setNN(2,6,6,51, 64,null);

        model.setBlock(block);

        TrainingConfig config = new DefaultTrainingConfig(new GGPCustomLoss("loss",-1))
                .optInitializer(new UniformInitializer(), Parameter.Type.WEIGHT)
                .addTrainingListeners(TrainingListener.Defaults.logging());
        Trainer trainer = model.newTrainer(config);
        trainer.initialize(new Shape( 1,1, 2, 6, 6));
        Shape shape=X.getShape();
        System.out.println("input"+shape);
        for (int i = 0; i < block.getChildren().size(); i++) {
            shape=block.getChildren().get(i).getValue().getOutputShapes(new Shape[]{shape})[0];
            System.out.println(shape);
        }
    }

    public static long getParameterNumber(){

        Model model = Model.newInstance("model");
        AbstractBlockGenerator cblock = new ConvolutionalBlockC();
        Block block = cblock.setNN(1,9,9,11, 64,null);

        model.setBlock(block);

        TrainingConfig config = new DefaultTrainingConfig(new GGPCustomLoss("loss",-1))
                .optInitializer(new UniformInitializer(), Parameter.Type.WEIGHT)
                .addTrainingListeners(TrainingListener.Defaults.logging());
        Trainer trainer = model.newTrainer(config);
        trainer.initialize(new Shape( 1,1, 2, 6, 6));
        long sum=0;
        for(int i = 0 ; i < model.getBlock().getParameters().size() ; i++){
            System.out.println(model.getBlock().getParameters().get(i).getKey());
            long s = model.getBlock().getParameters().get(i).getValue().getArray().getShape().size();
            System.out.println(s);
            System.out.println(sum+=s);
        }
        return sum;
    }


}