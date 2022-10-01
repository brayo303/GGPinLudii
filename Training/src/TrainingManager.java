import Block.NeuralNetworkManager;

import DataStructure.TrainIter;
import TestUtil.Debug;
import customWrapper.CustomGameWrapper;
import game.Game;
import other.AI;
import other.GameLoader;
import other.context.Context;
import other.move.Move;
import other.trial.Trial;

import java.io.*;
import java.util.*;

public class TrainingManager {

    private Game game;
    private NeuralNetworkManager curnet;
    final List<AI> agents;
    private LinkedList <LinkedList<TrainIter>> history;
    private int outShape;
    private int iteration;
    private int dimZ,dimX,dimY;
    //fdp
    //private int itdeb;


    public TrainingManager(){

        agents = new ArrayList<AI>();
        if(Configuration.option.length()!=0) {
            // load and create game
            String[] split = Configuration.option.split(",");
            List<String> list = new LinkedList<>();
            for(int i = 0 ; i < split.length ; i++){
                list.add(split[i]);
                System.out.println(split[i]);
            }
            game= GameLoader.loadGameFromName(Configuration.gameName, list);
        }else{
            game= GameLoader.loadGameFromName(Configuration.gameName);
        }
        CustomGameWrapper wrapper = new CustomGameWrapper(game);

        outShape = wrapper.numDistinctActions();
        dimZ = wrapper.stateTensorsShape()[0];
        dimY = wrapper.stateTensorsShape()[1];
        dimX = wrapper.stateTensorsShape()[2];
        if(Configuration.startIteration==0) {
            curnet = new NeuralNetworkManager(null
                    ,Configuration.startIteration
                    ,dimZ
                    ,dimY
                    ,dimX
                    ,outShape,
                    Configuration.learningRate,
                    Configuration.argsEpoch,
                    Configuration.miniBatchSize,
                    Configuration.dropout,
                    Configuration.optimizerType,
                    Configuration.argsDepthOrFilterNum,
                    Configuration.architechture,
                    Configuration.feature);
            history = new LinkedList<LinkedList<TrainIter>>();
            curnet.save(Configuration.path,0);
            saveConfiguration(game);
        }
        else{

            curnet = new NeuralNetworkManager(Configuration.path,
                    Configuration.startIteration
                    ,dimZ
                    ,dimY
                    ,dimX
                    ,outShape,
                    Configuration.learningRate,
                    Configuration.argsEpoch,
                    Configuration.miniBatchSize,
                    Configuration.dropout,
                    Configuration.optimizerType,
                    Configuration.argsDepthOrFilterNum,
                    Configuration.architechture,
                    Configuration.feature);
            loadHistory("history",Configuration.startIteration-1);
            curnet.debugWeight();

            //curnet.debugWeight();
        }
        agents.add(null);
        agents.add(new TrainAgent(curnet,true));
        agents.add(new TrainAgent(curnet,true));
        for (int p = 1; p < agents.size(); ++p)
        {
            agents.get(p).initAI(game, p);
        }
    }



    public LinkedList<TrainIter> selfPlay(int iteration){
        LinkedList<TrainIter> result = new LinkedList<>();
        Trial trial = new Trial(game);
        Context context = new Context(game, trial);
        game.start(context);


        int episodeIteration=0;
        // keep going until the game is over
        while (!context.trial().over())
        {
            //System.out.println("in");
            final int mover = context.state().mover();

            TrainAgent agent = (TrainAgent) agents.get(mover);
            //fdp
           //StatePrinter.filename="D:/GGPSkripsi/Training/logdebug/"+itdeb+"."+iteration+"."+episodeIteration+".txt";
            //System.out.println(move);
            //            StatePrinter.printlnpiece(context);
//            System.out.println(episodeIteration);
            agent.doMCTS(context);



            if(episodeIteration<Configuration.tempThreshold) {
                result.add(agent.getActionProbability(game,1));

            }else{
                result.add(agent.getActionProbability(game,0));
                //System.out.println("a");
            }

            final Move move = agent.selectAction(game, context, 0.2, -1, -1);

//            for(int i = 0 ; i <agent.getNodeNow().getChildNodeList().size() ; i++){
//                System.out.println("m"+agent.getNodeNow().getChildNodeList().get(i).getMoveFromParent()+agent.getNodeNow().getChildNodeList().get(i).getN());
//            }
//            StatePrinter.printlnpiece(context);
            //System.out.println("move"+move);

            game.apply(context, move);

            episodeIteration++;
        }

        Iterator<TrainIter> it = result.iterator();
        int winner = context.trial().status().winner();
        //StatePrinter.printState2D(context);
        System.out.println("winner:"+winner);
        if(winner!=0) {
            while (it.hasNext()) {
                TrainIter cur = it.next();
                //System.out.println(cur.getCurrentPlayer());
                cur.setWin(cur.getCurrentPlayer()==winner?1.0f:-1.0f);
                //System.out.println(cur);
                //temp.add(cur);

            }

        }else{
            while (it.hasNext()) {
                TrainIter cur = it.next();
                //System.out.println(cur.getCurrentPlayer());
                cur.setWin(0);
                //System.out.println(cur);
                //temp.add(cur);
            }
        }
        return result;
    }




    public void learn(){
        ArrayList<float[]> prevdiff[] = null;
        for(iteration = Configuration.startIteration ; iteration < Configuration.argsTrainIter ; iteration ++){

            //fdp
            //itdeb=iteration;

            //toggle self play on
            for(int i = 1 ; i < agents.size() ; i++){
                ((TrainAgent)agents.get(i)).setSelfPlay(true);
            }
            LinkedList<TrainIter> curList = new LinkedList<>();
            for(int j = 0 ; j < Configuration.argsNumSelfPlay ; j++){
                if(j==0){
                    Debug.flag=false;
                }else{
                    Debug.flag=false;
                }



                curList.addAll(selfPlay(j));

                //debug();



                System.out.println("selfplayit-"+j);

            }
            //StatePrinter.filename="D:/GGPSkripsi/Training/logdebug/dump";
            if(history.size()>Configuration.argsHistoryIter){
                history.poll();
                history.add(curList);
            }else{
                history.add(curList);
            }
            //  ensuring weight of nn
            curnet.train(history,Configuration.argsEpoch);
            logLoss(curnet.getLoss());


            //  ensuring weight of nn
            //prevNet.debugWeight();
            if(iteration>=Configuration.skipEval) {
                // ensuring weight of nn
                //curnet.debugWeight();
                NeuralNetworkManager prevNet = new NeuralNetworkManager(Configuration.path
                        ,iteration
                        ,dimZ
                        ,dimY
                        ,dimX
                        ,outShape,
                        Configuration.learningRate,
                        Configuration.argsEpoch,
                        Configuration.miniBatchSize,
                        Configuration.dropout,
                        Configuration.optimizerType,
                        Configuration.argsDepthOrFilterNum,
                        Configuration.architechture,
                        Configuration.feature);
                int competitionResult[][] = new int[2][];
                CompetitionManager competition = new CompetitionManager(game);
                competitionResult[0] = competition.compete(new TrainAgent(prevNet,false,true), new TrainAgent(curnet,false,true), Configuration.argsEvalCompetition / 2, "old network win", "new network win", "draw");
                competitionResult[1] = competition.compete(new TrainAgent(curnet,false,true), new TrainAgent(prevNet,false,true), Configuration.argsEvalCompetition / 2, "new network win", "old network win", "draw");
                float result = (float) (competitionResult[0][1] + competitionResult[1][0]) / (float) (competitionResult[0][1] + competitionResult[1][0] + competitionResult[0][0] + competitionResult[1][1]);
                logWinner(competitionResult[0][0] + competitionResult[1][1], competitionResult[0][1] + competitionResult[1][0]);
                System.out.println("past:" + ((int) (competitionResult[0][0] + competitionResult[1][1])));
                System.out.println("current:" + ((int) (competitionResult[1][0] + competitionResult[0][1])));
                if (result < Configuration.winThreshold || (competitionResult[0][1] + competitionResult[1][0] == 0&& !Configuration.drawAccept) ) {
                    System.out.print("network doesnt improve");
                    prevNet.save(Configuration.path,iteration+1);
                } else {
                    System.out.print("improve");
                    curnet.save(Configuration.path,iteration+1);
                }
                curnet.closeModel();
                prevNet.closeModel();
                curnet =  new NeuralNetworkManager(Configuration.path
                        ,iteration+1
                        ,dimZ
                        ,dimY
                        ,dimX
                        ,outShape,
                        Configuration.learningRate,
                        Configuration.argsEpoch,
                        Configuration.miniBatchSize,
                        Configuration.dropout,
                        Configuration.optimizerType,
                        Configuration.argsDepthOrFilterNum,
                        Configuration.architechture,
                        Configuration.feature);
            }else{
                System.out.print("no eval");
                curnet.save(Configuration.path,iteration+1);
            }
            ((TrainAgent)agents.get(1)).setNN(curnet);
            ((TrainAgent)agents.get(2)).setNN(curnet);
            saveHistory("history",iteration);

            curnet.debuglast(curnet.getLastWeight(),prevdiff);
            prevdiff= curnet.getLastWeight();
        }




    }





    /**
     * Method to manage the configuration of training
     * @param args arguments with format according to TrainAgent
     */
    public static void configure(String[] args){
        Configuration.gameName = args[0];
        Configuration.option = args[1];
        Configuration.path = args[2];

        Configuration.architechture = args[3].charAt(0);
        Configuration.learningRate = Float.parseFloat(args[4]);
        Configuration.argsDepthOrFilterNum = Integer.parseInt(args[5]);
        Configuration.miniBatchSize = Integer.parseInt(args[6]);
        Configuration.startIteration = Integer.parseInt(args[7]);
        Configuration.argsTrainIter = Integer.parseInt(args[8]);
        Configuration.argsNumSelfPlay = Integer.parseInt(args[9]);
        Configuration.argsHistoryIter = Integer.parseInt(args[10]);
        Configuration.argsEpoch = Integer.parseInt(args[11]);
        Configuration.argsEvalCompetition = Integer.parseInt(args[12]);
        Configuration.mcts_it = Integer.parseInt(args[13]);
        Configuration.mcts_it_eval = Integer.parseInt(args[14]);
        Configuration.winThreshold = Float.parseFloat(args[15]);
        Configuration.tempThreshold = Integer.parseInt(args[16]);
        Configuration.skipEval = Integer.parseInt(args[17]);
        Configuration.optimizerType = args[18].charAt(0);

        if(args[19].length()!=0) {
            String[] split = args[19].split(",");
            int[] val = new int[split.length];
            for (int i = 0; i < split.length; i++) {
                val[i] = Integer.parseInt(split[i]);
            }

            Configuration.feature=val;
        }else {
            Configuration.feature = null;
        }

        if(args[20].length()>0) {
            Configuration.epsilon = Float.parseFloat(args[20]);
        }else{
            Configuration.epsilon = null;
        }

        if(args[21].length()>0) {
            Configuration.alpha = Double.parseDouble(args[21]);
        }else{
            Configuration.alpha = null;
        }

        Configuration.drawAccept = Boolean.parseBoolean(args[22]);
        if(args.length>23) {
            Configuration.dropout = Float.parseFloat(args[23]);
        }else{
            Configuration.dropout=null;
        }
    }

    public void saveHistory(String prefix,Integer it){

        try {

            FileOutputStream fout = new FileOutputStream(Configuration.path+prefix+"-"+it);
            ObjectOutputStream out = new ObjectOutputStream(fout);
            out.writeObject(history);
            out.close();
        } catch (IOException e) {
            e.printStackTrace();
        }

    }


    public void logWinner(int ... x) {
        try {
            BufferedWriter writer = new BufferedWriter(new FileWriter(Configuration.path+"winningLogs.txt",true));
            for(int i = 0 ; i < x.length ; i++) {
                writer.write("player"+i+":"+x[i]+",");
            }
            writer.newLine();
            writer.close();
        } catch (FileNotFoundException e) {
            // TODO Auto-generated catch block
            e.printStackTrace();
        } catch (IOException e) {
            // TODO Auto-generated catch block
            e.printStackTrace();
        }
    }
    public void logLoss(String loss) {
        try {
            BufferedWriter writer = new BufferedWriter(new FileWriter(Configuration.path+"winningLogs.txt",true));
            writer.write(iteration+"");
            writer.newLine();
            writer.write("loss :"+loss);
            writer.newLine();
            writer.close();
        } catch (FileNotFoundException e) {
            // TODO Auto-generated catch block
            e.printStackTrace();
        } catch (IOException e) {
            // TODO Auto-generated catch block
            e.printStackTrace();
        }
    }

    public static void logArgs(){
        try {
            BufferedWriter writer = new BufferedWriter(new FileWriter(Configuration.path+"configuration.txt",true));
            writer.write("Game Name:"+Configuration.gameName);
            writer.newLine();
            writer.write("option:"+Configuration.option);
            writer.newLine();
            if(Configuration.architechture=='c') {
                writer.write("architechture: convolutional");
            }
            else if(Configuration.architechture=='d') {
                writer.write("architechture: dense");
            }
            else if(Configuration.architechture=='b') {
                writer.write("architechture: convolutionalb");
            }
            else if(Configuration.architechture=='a') {
                writer.write("architechture: convc");
            }else if(Configuration.architechture=='g'){
                writer.write("architechture: suragnair dkk");
            }
            writer.newLine();
            writer.write("Path:"+Configuration.path);
            writer.newLine();
            writer.write("Learning Rate:"+Configuration.learningRate);
            writer.newLine();
            if(Configuration.architechture=='c') {
                writer.write("Start Filter Number:" + Configuration.argsDepthOrFilterNum);
            }else if(Configuration.architechture=='d'){
                writer.write("Depth:" + Configuration.argsDepthOrFilterNum);
            }
            writer.newLine();
            writer.write("epoch:"+Configuration.argsEpoch);
            writer.newLine();
            writer.write("miniBatchSize:"+Configuration.miniBatchSize);
            writer.newLine();
            writer.write("Train Iter:"+Configuration.argsTrainIter);
            writer.newLine();
            writer.write("Number of selfplay game:"+ Configuration.argsNumSelfPlay);
            writer.newLine();
            writer.write("History Iteration:" + Configuration.argsHistoryIter);
            writer.newLine();
            writer.write("Evaluation game self play:" + Configuration.argsEvalCompetition);
            writer.newLine();
            writer.write("MCTS Iteration:" + Configuration.mcts_it);
            writer.newLine();
            writer.write("MCTS eval"+Configuration.mcts_it_eval);
            writer.newLine();
            writer.write("Win Threshold:"+Configuration.winThreshold);
            writer.newLine();
            writer.write("temperature Threshold:"+Configuration.tempThreshold);
            writer.newLine();
            writer.write("skipEval:"+Configuration.skipEval);
            writer.newLine();

            if(Configuration.optimizerType=='s'){
                writer.write("Optimizer: SGD");
                writer.newLine();
            }else{
                writer.write("Optimizer: Adam");
                writer.newLine();
            }
            if(Configuration.dropout!=null) {
                writer.write("dropout rate:" + Configuration.dropout);
            }else{
                writer.write("nodrop");
            }
            writer.newLine();
            if(Configuration.alpha!=null&&Configuration.epsilon!=null){
                writer.write("alpha:"+Configuration.alpha+" epsilon:"+Configuration.epsilon);
                writer.newLine();;
            }
            if(Configuration.feature!=null){
                writer.write("feature:");
                for(int i = 0 ; i < Configuration.feature.length ; i++){
                    writer.write(Configuration.feature[i]+",");
                }
            }
            writer.newLine();
            if(!Configuration.drawAccept) {
                writer.write("cant draw");
            }



            writer.close();
        } catch (FileNotFoundException e) {
            // TODO Auto-generated catch block
            e.printStackTrace();
        } catch (IOException e) {
            // TODO Auto-generated catch block
            e.printStackTrace();
        }
    }

    public void loadHistory(String prefix,Integer it){

            try {

                FileInputStream fin = new FileInputStream(Configuration.path + prefix + "-" + it);
                ObjectInputStream in = new ObjectInputStream(fin);
                history = (LinkedList<LinkedList<TrainIter>>)  in.readObject();


                //checker must one
                //for(int i = 0 ; i <history.get(1).getTensorAction().length ; i++){
                //                if(history.get(1).getTensorAction()[i]>0){
                //                    System.out.print(history.get(1).getTensorAction()[i]+" ");
                //                }
                //            }
            } catch (IOException | ClassNotFoundException e) {
                System.out.println("error (load)ing history");
                history = new LinkedList<LinkedList<TrainIter>>();
            }

    }

    public void saveConfiguration(Game game){




        try {
            FileWriter wr = new FileWriter(Configuration.path+"/configure.txt");
            BufferedWriter br = new BufferedWriter(wr);
            br.write(Configuration.mcts_it+"");
            br.newLine();
            if(Configuration.dropout==null){
                br.write("-");
            }else{
                br.write(Configuration.dropout+"");
            }
            br.newLine();
            br.write(Configuration.argsDepthOrFilterNum+"");
            br.newLine();
            br.write(Configuration.architechture);
            br.newLine();
            if(Configuration.feature==null){
                br.write("-");
            }
            else{
                for(int i = 0 ; i < Configuration.feature.length ; i++){
                    br.write(Configuration.feature[i]);
                    if(i<Configuration.feature.length-1){
                        br.write(",");
                    }
                }
            }
            br.close();

        } catch (FileNotFoundException e) {
            e.printStackTrace();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }


   public static void  main(String[]args){
       try {
           configure(args);

       }catch(Exception e){
           System.out.println(e);
           System.out.println("Using default value");
       }
       logArgs();
       TrainingManager train = new TrainingManager();
       train.learn();


   }
}
