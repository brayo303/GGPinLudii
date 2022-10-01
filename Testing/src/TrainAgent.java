

import Block.NeuralNetworkManager;
import DataStructure.NodeWithWrapper;
import DataStructure.TrainIter;
import customWrapper.CustomGameWrapper;
import customWrapper.CustomStateWrapper;
import game.Game;
import main.collections.FastArrayList;
import other.AI;
import other.context.Context;
import other.move.Move;

import java.io.*;
import java.util.Iterator;
import java.util.LinkedList;
import java.util.List;
import java.util.concurrent.ThreadLocalRandom;


/**
 * TrainAgent is a class that doing specially designed to do training
 */
public class TrainAgent extends AI
{

    //decalre tree object with node datastructure
    private NodeWithWrapper node;
    private NodeWithWrapper root;

    //declare object for helping the training
    private NeuralNetworkManager network;

    // declare variable for storing current player
    private int player;

    // TBD
    //protected Move lastReturnedMove;
    private boolean isSelfPlay;
    private boolean noReset;
    private boolean generateNetwork;
    private Move nextMove;
    private boolean firstPlay;
    private String path;



    public TrainAgent(){
        friendlyName = "TrainAgents";
        generateNetwork=true;
        System.out.println("this is test mode");

    }


    /**
     * Contructor of train agent
     * @param network neural network  used
     */
    public TrainAgent(NeuralNetworkManager network, boolean isSelfPlay)
    {
        friendlyName = "TrainAgent";
        this.network= network;
        this.isSelfPlay = isSelfPlay;
        this.generateNetwork=false;

    }

    public TrainAgent(NeuralNetworkManager network, boolean isSelfPlay, boolean noReset)
    {
        friendlyName = "TrainAgent";
        this.network= network;
        this.isSelfPlay = isSelfPlay;
        this.generateNetwork=false;
        this.noReset=noReset;

    }



    /**
     * Setter of neural network object
     * @param network new neural network for train agent
     */
    public void setNN(NeuralNetworkManager network){

        this.network=network;
    }

    public void resetNode(){
        //System.out.println("reset");
        //StatePrinter.printState2D(root.getContext());
        this.node = root;
    }

    public void setSelfPlay(boolean isSelfPlay){
        this.isSelfPlay=isSelfPlay;
    }


    public NodeWithWrapper getRoot(){
        return root;
    }

    public NodeWithWrapper getNodeNow(){
        return node;
    }

    @Override
    public Move selectAction
    (
            final Game game,
            final Context context,
            final double maxSeconds,
            final int maxIterations,
            final int maxDepth
    )
    {
        //check is current selfplay phase
        if(isSelfPlay==false) {
            if(generateNetwork==true) {
                getConfiguration(game);
                friendlyName = "TrainAgents"+Configuration.mcts_it;
                CustomGameWrapper wrapper = new CustomGameWrapper(game);
                int outShape = wrapper.numDistinctActions();
                network =new NeuralNetworkManager(Configuration.path,100,wrapper.stateTensorsShape()[0]
                        ,wrapper.stateTensorsShape()[1]
                        ,wrapper.stateTensorsShape()[2]
                        ,outShape,
                        1,
                        1,
                        1,
                        Configuration.dropout,
                        Configuration.optimizerType,
                        Configuration.argsDepthOrFilterNum,
                        Configuration.architechture,
                        Configuration.feature);
                this.isSelfPlay = false;
            }
            // Initiating the mcts
            MCTSDeepLearning mcts = new MCTSDeepLearning(network);
            if(noReset==false) {
                //System.out.println("test-mode");
                // Initiating the tree
                node = new NodeWithWrapper(null, context, player, null);
                // doing mcts with iteration acording to Configuration class
                for (int i = 0; i < Configuration.mcts_it; i++) {
                    // do one iteration of mcts using current tree representation
                    mcts.search(node);
                }
                Move bestMove = mcts.getBestAction(node);
                // return the best move so far
                return bestMove;
            }else{
                Move lastMove = context.trial().lastMove();
                if(node == null&&root==null){
                    System.out.println("root");
                    if(player==2){

                        //System.out.println("parent context");
                        //StatePrinter.printState2D(parentContext);
                        node = new NodeWithWrapper(null, new Context(context), player, null);
                        firstPlay=true;
                    }else{
                        node = new NodeWithWrapper(null, new Context(context), player, null);
                        firstPlay=true;
                    }
                    root = node;
                   // System.out.println("root");
                    //StatePrinter.printState2D(root.getContext());
                }

                //System.out.println(context.state().mover()+","+firstPlay);
                if(firstPlay==false){

                    //System.out.println("not root");

                    NodeWithWrapper nextNode = node.searchChildNode(lastMove);
                    //System.out.println("statenew");
                    //StatePrinter.printState2D(node.getContext());
                    if (nextNode == null) {
                        mcts.forceExpand(node, lastMove);
                        nextNode = node.searchChildNode(lastMove);
                    }
                    node = nextNode;

                }

                for (int i = 0; i < Configuration.mcts_it_eval; i++) {
                    // do one iteration of mcts using current tree representation
                    mcts.search(node);
//                    System.out.println(node.getPlayer());
                    //getNetwork();
//                    System.out.println(node.getParent());
                }

                node = mcts.getBestNode(node);
                //System.out.println(node);
                this.firstPlay=false;
                return node.getMoveFromParent();

            }

        }else{
            // return random move
            //System.out.println("ss");
            return nextMove;
        }

    }

    public void doMCTS(Context context){
        MCTSDeepLearning mcts = new MCTSDeepLearning(network,Configuration.epsilon,Configuration.alpha);
        // Initiating the tree
        node = new NodeWithWrapper(null, context, player, null);

        // doing mcts with iteration acording to Configuration class
        for(int i = 0 ; i < Configuration.mcts_it ; i++) {
            //fdp
            //StatePrinter.logln(i+"");
            // do one iteration of mcts using current tree representation
            mcts.search(node);
        }
    }

    /**
     * Force MCTS for player 2 in competition Manager
     */
    public void forceMCTS(){
        MCTSDeepLearning mcts = new MCTSDeepLearning(network);

        for(int i = 0 ; i < Configuration.mcts_it_eval ; i++) {
            //fdp
            //StatePrinter.logln(i+"");
            // do one iteration of mcts using current tree representation
            mcts.search(node);
        }
    }


    /**
     * Convert current state to stateTensor
     * Calculate the probability of one move and map it into move tensor
     * @param game current game object of Ludii
     * @return TrainIter object that used to store history data
     */
    public TrainIter getActionProbability(Game game, int temp){
        // initiate new gameWrapper object
        CustomGameWrapper gameWrapper = new CustomGameWrapper(game);
        // initiate new stateWrapper object
        CustomStateWrapper stateWrapper = new CustomStateWrapper(gameWrapper,node.getContext());
        // get all child node that already represented as tree
        FastArrayList<NodeWithWrapper> childNodeList = node.getChildNodeList();
        // store state tensor value got from the state wrapper
        float [][][] stateTensor = stateWrapper.toTensor();
        // store action tensor value got from the game wrapper
        float [] actionTensor = new float[gameWrapper.numDistinctActions()];
        //check temp
        if(temp==1) {
            // initialize variable for storing sum of all child node visit count
            int sum = 0;
            //fdp
//            StatePrinter.logState2D(node.getContext());
            // sigma loop of child node visit count
            for (int i = 0; i < childNodeList.size(); i++) {
                // summing the sum
                sum += childNodeList.get(i).getN();
                //fdp
                //StatePrinter.logln(childNodeList.get(i).getMoveFromParent()+" "+childNodeList.get(i).getN() + " " + childNodeList.get(i).getP() + " "+childNodeList.get(i).getQ());
            }
            //find next move random
            int randNumber = ThreadLocalRandom.current().nextInt(sum);
            //sum for random
            int randSum = 0;
            // doing mapping of all move from parent in child node
            for (int i = 0; i < childNodeList.size(); i++) {
                //mapping with the help of gamewrapper and store the value of N(v_i)/Sigma_j=0...n(N(v_(j)))
                actionTensor[gameWrapper.moveToInt(childNodeList.get(i).getMoveFromParent())] = (float) childNodeList.get(i).getN() / sum;
                //find next
                randSum+= childNodeList.get(i).getN();
                if(randSum>randNumber){
                    nextMove = childNodeList.get(i).getMoveFromParent();
                    randNumber=sum;
                }
            }
        }else{
            // make temporary maximum variable
            int curentmaximum = -1;
            // make list for storing index with maximum value
            LinkedList<Integer> idxmaximum = new LinkedList<Integer>();
            // loop all childnode
            for(int i = 0 ; i < childNodeList.size() ; i++){
                // check wether current equal or more than temporary maximum
                if(childNodeList.get(i).getN()==curentmaximum){
                    // if current maximum = temporary maximum it means we add the index to the list
                    idxmaximum.add(i);
                }else if(childNodeList.get(i).getN()>curentmaximum){
                    // if current maximum more temporary maximum it means we add the index to the list
                    curentmaximum=childNodeList.get(i).getN();
                    // and clear the list
                    idxmaximum.clear();
                    // then add the list with the new index
                    idxmaximum.add(i);
                }
            }
            if(!childNodeList.isEmpty()) {
                // create random
                //Random rand = new Random();
                // get index maximum
                int choosedindex = idxmaximum.get(ThreadLocalRandom.current().nextInt(idxmaximum.size()));
                // set the choosed node to one
                actionTensor[gameWrapper.moveToInt(childNodeList.get(choosedindex).getMoveFromParent())] = 1.0f;
                //set nextMove
                nextMove = childNodeList.get(choosedindex).getMoveFromParent();
            }
        }
        // save as TrainIter object
        //System.out.println("player:"+player);
        TrainIter trainIter = new TrainIter(stateTensor,actionTensor,player);

        //debug
//        if(node.getContext().trial().over()){
//            trainIter.setDebug(node.getContext());
//        }

        //return the value
        return trainIter;
    }


    //    TBD
//    public Move lastReturnedMove()
//    {
//        return lastReturnedMove;
//    }

    @Override
    public void initAI(final Game game, final int playerID)
    {
        System.setProperty("java.library.path","C:/Users/ROG/.djl.ai/pytorch/1.9.1-cpu-win-x86_64");
        System.setProperty("PYTORCH_VERSION","1.9.1");
        System.setProperty("PYTORCH_FLAVOR","cpu");
        this.player = playerID;




    }




    public void initAI(final Game game, final int playerID, NodeWithWrapper inputNode, boolean firstPlay)
    {
        this.firstPlay=firstPlay;
        node = inputNode;
        root = inputNode;
        initAI(game,playerID);
    }

    //-------------------------------------------------------------------------
    //fdp

    public void getConfiguration(Game game){
        File directory = new File("./");

        List<String> opt = game.getOptions();
        Iterator<String> it = opt.iterator();
        String option = "";
        while(it.hasNext()){
            String[] temp = it.next().split("/");
            for(int i = 0 ; i < temp.length ; i++){
                option+=temp[i];
            }
        }
        String path = directory.getAbsolutePath()+game.name()+option;
        this.path=path;
        try {
            FileReader fr = new FileReader(path+"/configure.txt");
            BufferedReader br = new BufferedReader(fr);
            Configuration.path = path;
            Configuration.mcts_it = Integer.parseInt(br.readLine());
            String dout = br.readLine();
            if(dout.charAt(0)!='-'){
                Configuration.dropout = Float.parseFloat(dout);
            }else{
                Configuration.dropout=null;
            }
            Configuration.argsDepthOrFilterNum = Integer.parseInt(br.readLine());
            Configuration.architechture = br.readLine().charAt(0);
            String[] features= br.readLine().split(",");
            if(features[0].charAt(0)!='-'){
                Configuration.feature = new int[features.length];
                for(int i = 0 ; i < features.length ; i++){
                    Configuration.feature[i]=Integer.parseInt(features[i]);
                }
            }else{
                Configuration.feature=null;
            }
            br.close();

        } catch (FileNotFoundException e) {
            e.printStackTrace();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public  void getNetwork(){
        network.debugWeight();
    }

    public void resetFirstPlay(){
        firstPlay=true;
    }


    @Override
    public String generateAnalysisReport()
    {
        return path;
    }


}





