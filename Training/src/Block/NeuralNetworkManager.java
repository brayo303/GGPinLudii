package Block;




import DataStructure.TrainIter;
import ai.djl.MalformedModelException;
import ai.djl.Model;

import ai.djl.inference.Predictor;
import ai.djl.metric.Metrics;
import ai.djl.ndarray.BaseNDManager;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.index.NDIndex;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.Block;
import ai.djl.training.DefaultTrainingConfig;
import ai.djl.training.EasyTrain;


import ai.djl.training.Trainer;
import ai.djl.training.dataset.ArrayDataset;


import ai.djl.training.listener.*;
import ai.djl.training.optimizer.Optimizer;
import ai.djl.training.tracker.Tracker;
import ai.djl.translate.Batchifier;
import ai.djl.translate.TranslateException;
import ai.djl.translate.Translator;
import ai.djl.translate.TranslatorContext;

import java.io.IOException;
import java.nio.file.Paths;
import java.util.*;


public class NeuralNetworkManager {

    private float learningrate ;
    private final int argsBatchSize ;
    private Float dropout;
    private Model model;
    private Block block;
    private DefaultTrainingConfig config;
    private Trainer trainer;
    private Predictor<float[][][],float[][]> predictor;
    private Shape shapein;
    private int selectFeatures[];
    Tracker lrt;
    Translator<float[][][], float[][]> translator;
    char optimizerType;
    int epoch;
    Integer episode;
    String path;
    boolean dropLast;

    public NeuralNetworkManager(String path, int episode, int inSizeZ, int inSizeY, int inSizeX, int outSize , float learningrate, int epoch, int argsBatchSize, Float dropout, char optimizerType, int depthOrFilterNum, char architechture, int[] selectFeatures){
        this.learningrate = learningrate;
        this.argsBatchSize = argsBatchSize;
        this.dropout = dropout;
        this.selectFeatures = selectFeatures;
        this.optimizerType=optimizerType;
        this.epoch = epoch;
        this.episode = episode;
        this.path = path;
        if(selectFeatures!=null){
            inSizeZ=selectFeatures.length;
        }


        initNetwork(inSizeZ, inSizeY, inSizeX, outSize, dropout, depthOrFilterNum, architechture);

    }

    public void initNetwork(int inSizeZ, int inSizeY,int inSizeX, int outSize, Float dropout, int startFilterNum,char architechture){

        model = Model.newInstance("model");



        lrt = Tracker.fixed(learningrate);
        //Optimizer optimizer= (new AdamCustom.Builder()).optBeginNumUpdate(numUpdate).optLearningRateTracker(lrt).build();
        Optimizer optimizer;
        if(optimizerType=='s') {
            optimizer = Optimizer.sgd().setLearningRateTracker(lrt).build();
        }else {
            optimizer = Optimizer.adam().optLearningRateTracker(lrt).build();

        }
        //Optimizer optimizer = (new SGDCustom.Builder()).setLearningRateTracker(lrt).build();

        this.config = new DefaultTrainingConfig(new GGPCustomLoss("loss",-1))
                .addTrainingListeners(TrainingListener.Defaults.logging())
                .optOptimizer(optimizer);

        //System.out.println(trainer.getEvaluators());



        AbstractBlockGenerator cblock = null;
        if(architechture=='c'){
            cblock = new ConvolutionalBlock();
            block = cblock.setNN(inSizeZ,inSizeY,inSizeX,outSize,startFilterNum, dropout);
            if(selectFeatures==null) {
                this.shapein = new Shape(1, 1, inSizeZ, inSizeY, inSizeX);
            }else{
                this.shapein = new Shape(1,1,selectFeatures.length,inSizeY, inSizeX);
            }
        }else if(architechture=='d'){
            cblock = new DenseBlock();
            System.out.println(inSizeZ);
            block = cblock.setNN(inSizeZ,inSizeY,inSizeX,outSize,startFilterNum, dropout);
            if(selectFeatures==null) {
                this.shapein = new Shape(1,inSizeZ* inSizeY*inSizeX);
            }else{
                this.shapein = new Shape(1,selectFeatures.length*inSizeY*inSizeX);
            }
        }else if(architechture=='b'){
            cblock = new ConvolutionalBlockB();
            block = cblock.setNN(inSizeZ,inSizeY,inSizeX,outSize,startFilterNum, dropout);
            if(selectFeatures==null) {
                this.shapein = new Shape(1, 1, inSizeZ, inSizeY, inSizeX);
            }else{
                this.shapein = new Shape(1,1,selectFeatures.length,inSizeY, inSizeX);
            }
        }else if(architechture=='a'){
            cblock = new ConvolutionalBlockC();
            block = cblock.setNN(inSizeZ,inSizeY,inSizeX,outSize,startFilterNum, dropout);
            if(selectFeatures==null) {
                this.shapein = new Shape(1, 1, inSizeZ, inSizeY, inSizeX);
            }else{
                this.shapein = new Shape(1,1,selectFeatures.length,inSizeY, inSizeX);
            }
        }else if(architechture=='g'){
            cblock = new GGPBlock();
            block = cblock.setNN(inSizeZ,inSizeY,inSizeX,outSize,startFilterNum, dropout);
            if(selectFeatures==null) {
                this.shapein = new Shape(1, 1, inSizeZ, inSizeY, inSizeX);
            }else{
                this.shapein = new Shape(1,1,selectFeatures.length,inSizeY, inSizeX);
            }
        }else{
            cblock = new NoMaxPoolBlock();
            block = cblock.setNN(inSizeZ,inSizeY,inSizeX,outSize,startFilterNum, dropout);
            if(selectFeatures==null) {
                this.shapein = new Shape(1, 1, inSizeZ, inSizeY, inSizeX);
            }else{
                this.shapein = new Shape(1,1,selectFeatures.length,inSizeY, inSizeX);
            }
        }
        if(cblock!=null) {
            dropLast = cblock.isDropLast();
        }
        model.setBlock(block);
        if(path!=null){
            load(path,episode);
        }
        trainer = model.newTrainer(config);
        trainer.initialize(shapein);
        trainer.setMetrics(new Metrics());



        //evaluatorMetrics = new HashMap<>();
        setTranslator();

    }

    public void resetTrainer(){
        System.out.println(trainer.getTrainingResult());

    }



    public void train(LinkedList<LinkedList<TrainIter>> trainIterLinkedList,int epoch){
        //initNetwork(trainIterLinkedList.peek().getTensorState().length,1,trainIterLinkedList.peek().getTensorAction().length);

        int sizeBatch = 0;
        Iterator<LinkedList<TrainIter>> ith =trainIterLinkedList.listIterator();
        while(ith.hasNext()){
            sizeBatch += ith.next().size();
        }



        int dimZ = trainIterLinkedList.getFirst().getFirst().getTensorState().length;
        int dimY = trainIterLinkedList.getFirst().getFirst().getTensorState()[0].length;
        int dimX = trainIterLinkedList.getFirst().getFirst().getTensorState()[0][0].length;
        NDArray arrayin;
        if(selectFeatures==null) {
            arrayin = model.getNDManager().create(new Shape(sizeBatch, 1, dimZ, dimY, dimX));
        }else{
            arrayin = model.getNDManager().create(new Shape(sizeBatch, 1, selectFeatures.length, dimY, dimX));
        }
        float out1[][] = new float[sizeBatch][];
        float out2[][] = new float[sizeBatch][];

        ith =trainIterLinkedList.listIterator();
        int cur =0 ;
        while(ith.hasNext()){

            Iterator<TrainIter> it = ith.next().listIterator();
            while(it.hasNext()) {
                TrainIter curIterData = it.next();
                float data[][][] = preprocess(curIterData.getTensorState());
                for (int i = 0; i < data.length; i++) {
                    for (int j = 0; j < data[i].length; j++) {
                        for (int k = 0; k < data[i][j].length; k++) {
                            //System.out.print(data[i][j][k]+" ");
                            arrayin.set(new NDIndex(cur, 0, i, j, k), data[i][j][k]);
                        }
                        //System.out.println();

                    }
                    //System.out.println("===");
                }
                out1[cur] = curIterData.getTensorAction();
                out2[cur] = new float[]{curIterData.getWin()};
                //System.out.println(curIterData);
                cur++;
            }


        }

        NDArray outa = model.getNDManager().create(out1);
        NDArray outb = model.getNDManager().create(out2);

//        System.out.println(Arrays.toString(out2[0]));
//        System.out.println(outb.get(0));
//        for(int i = 0 ; i < outb.size() ; i++){
//            System.out.print(outb.get(i).toFloatArray()[0]+" ");
//        }
       // System.out.println(outb);
        System.out.println(arrayin);
        //System.out.println(dropLast);
        ArrayDataset dataset = new ArrayDataset.Builder()
                .setData(arrayin) // set the features
                .optLabels(outa,outb) // set the labels
                .setSampling(argsBatchSize, true,dropLast)// set the batch size and random sampling
                .build();

//        for(int i = 0 ; i < dataset.size(); i++){
//            System.out.println(dataset.get(manager,i).getData().get(0));
//        }





        try {
            fit(dataset);

        } catch (TranslateException e) {
            e.printStackTrace();
        } catch (IOException e) {
            e.printStackTrace();
        }





    }









    //for debug purpose
    public void debugWeight(){

        System.out.print(block.getParameters().valueAt(0).getArray().get(0)+" ");

    }

    public  ArrayList<float[]>[] getLastWeight(){
        ArrayList<float[]> last[] = new ArrayList[block.getParameters().size()] ;

        for(int j = 0 ; j < block.getParameters().size(); j++) {
                last[j]= new ArrayList<>();

                for(int i = 0 ; i < block.getParameters().valueAt(j).getArray().size(0) ; i++) {

                    last[j].add(block.getParameters().valueAt(j).getArray().get(i).toFloatArray());
                }



        }
        return last;
    }

    public void debuglast( ArrayList<float[]> [] cur ,  ArrayList<float[]>[] prev){
        if(prev!=null) {
            boolean diff = false;
            int sum = 0;
            int lastidx = 0;
            int lastj = 0;
            int lastk=0;
            for(int i = 0 ; i< prev.length ; i++) {
                for(int j = 0 ; j < prev[i].size() ; j++){
                    for(int k = 0; k < prev[i].get(j).length ; k++){
                        if(cur[i].get(j)[k]!=prev[i].get(j)[k]){
                            diff=true;
                            sum++;
                            lastidx=i;
                            lastj = j;
                            lastk = k;
                        }
                    }
                }
            }
            if (diff) {
                System.out.println("difference in:"+sum+" ex: "+cur[lastidx].get(lastj)[lastk]+" "+prev[lastidx].get(lastj)[lastk]+" "+lastidx+":"+lastj+":"+lastk);
            } else {
                System.out.println("nodiffweight");
            }
        }


    }

    public boolean debugTest( ArrayList<float[]> [] cur ,  ArrayList<float[]>[] prev){
        if(prev!=null) {
            boolean diff = false;
            int sum = 0;
            int lastidx = 0;
            int lastj = 0;
            int lastk=0;
            for(int i = 0 ; i< prev.length ; i++) {
                for(int j = 0 ; j < prev[i].size() ; j++){
                    for(int k = 0; k < prev[i].get(j).length ; k++){
                        if(cur[i].get(j)[k]!=prev[i].get(j)[k]){
                            diff=true;
                            sum++;
                            lastidx=i;
                            lastj = j;
                            lastk = k;
                        }
                    }
                }
            }
            if (diff) {
                return true;
            } else {
                return false;
            }
        }
        return false;


    }







    public String getLoss(){
        return trainer.getTrainingResult().getTrainLoss()+"";
    }

    public void debugMemory(NDManager manager){
        ((BaseNDManager)manager).debugDump(0);
    }


    public void fit(ArrayDataset dataset) throws TranslateException, IOException {




        EasyTrain.fit(trainer,epoch,dataset,null);


    }



    public void save(String path, int episode) {



        try {
            model.setProperty("Epoch", String.valueOf(episode));
            model.save(Paths.get(path),"model");
        } catch (IOException e) {
            e.printStackTrace();
        }
    }



    public void closeModel(){
        trainer.close();
        model.close();
    }



    public void load(String path,int episode){

        try {

            Map options  = new HashMap<String,Integer>();
            options.put("epoch",episode);
            model.load(Paths.get(path),"model", options);
//            if(optimizerType=='a') {
//                Optimizer optimizer = Optimizer.adam().optBeginNumUpdate(epoch*episode).optLearningRateTracker(lrt).build();
//                config.optOptimizer(optimizer);
//            }
//            if(trainer!=null){
//                trainer.close();
//                trainer = model.newTrainer(config);
//                trainer.initialize(shapein);
//                trainer.setMetrics(new Metrics());
//            }
        } catch (IOException e) {
            e.printStackTrace();
        } catch (MalformedModelException e) {
            e.printStackTrace();
        }

    }

    public void setTranslator(){

        translator = new Translator<float[][][],float[][]>() {

            @Override
            public NDList processInput(TranslatorContext ctx, float [][][] in) {
                float input[][][] = preprocess(in);
                NDArray arrayInput =  ctx.getNDManager().create(new Shape(1,1,input.length,input[0].length,input[0][0].length));
                for(int i = 0 ; i < input.length ; i++) {
                    for(int j = 0 ; j <input[i].length ; j++){
                        for(int k=0 ; k <input[i][j].length; k++){

                            arrayInput.set(new NDIndex(0,0,i,j,k),input[i][j][k]);
                        }

                    }
                }

                NDList inputList = new NDList(arrayInput);


                return inputList;
            }

            @Override
            public float[][] processOutput(TranslatorContext ctx, NDList forwardPropResult) {
                float result [][] = new float[2][];
                //System.out.println(forwardPropResult.get(0));
                result[0]= forwardPropResult.get(0).softmax(-1).toFloatArray();
                result[1]= forwardPropResult.get(1).toFloatArray();
                return result;
            }

            @Override
            public Batchifier getBatchifier() {
                return null;
            }
        };

    }

    public float[][] forward(float data[][][]){
        //setTranslator();
        predictor = model.newPredictor(translator);
        try {
            float[][] result= predictor.predict(data);
            predictor.close();
            return result;
        } catch (TranslateException e) {
            e.printStackTrace();
        }
        predictor.close();
        return null;

    }


    float[][][] preprocess(float input[][][]){
        if(selectFeatures!=null) {
            float[][][] prepin = new float[selectFeatures.length][][];
            for (int i = 0; i < selectFeatures.length; i++) {
                prepin[i] = input[selectFeatures[i]];
            }
            return prepin;
        }else {
            return input;
        }

    }





}


