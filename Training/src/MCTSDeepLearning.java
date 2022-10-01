import Block.NeuralNetworkManager;
import DataStructure.NodeWithWrapper;
import jsat.distributions.multivariate.Dirichlet;
import jsat.linear.DenseVector;
import jsat.linear.Vec;
import main.collections.FastArrayList;
import other.RankUtils;
import other.context.Context;
import other.move.Move;
import search.mcts.MCTS;

import java.util.Random;
import java.util.concurrent.ThreadLocalRandom;

public class MCTSDeepLearning {




    // uct constant for maintaining expansion and exploitation
    private final double c_uct=1;
    // number of player which play the game
    private int playerNum = 2;
    // declare object for neural network used by MCTS
    private NeuralNetworkManager network;

    private Float epsilon;
    private Double alpha;

    public MCTSDeepLearning(NeuralNetworkManager network){
        this.network = network;
        epsilon=null;
        alpha =null;

    }
    public MCTSDeepLearning(NeuralNetworkManager network,float epsilon , Double alpha){
        this.network = network;
        this.epsilon=epsilon;
        this.alpha =alpha;

    }

    public int nextPlayer(int player) {
        return (((player-1)+1)%playerNum)+1;
    }

    public Move getNextUnexpandedMove(NodeWithWrapper node){
        if(!node.getUnexpandedMove().isEmpty()){
            return node.getUnexpandedMove().iterator().next().move;
        }
        return null;
    }

    public void expand(NodeWithWrapper node) {
        // create new context
        Context context = new Context(node.getContext());
        //getNextMove
        Move nextMove = getNextUnexpandedMove(node);
        // remove next move
        node.getUnexpandedMove().remove(new MCTS.MoveKey(nextMove,0));
        // do move to the context
        context.game().apply(context, nextMove);
        // make new node after applying the move to the context
        NodeWithWrapper newnode = new NodeWithWrapper(node,context,nextPlayer(node.getPlayer()),nextMove);
        // add node to the tree
        node.addChild(newnode);


    }

    /**
     * Method to do playout
     * @param node starting node of playout
     * @return value produced by neural network
     */
    public double playoutPolicy(NodeWithWrapper node) {

//        if(node.getContext().trial().over()){
//            System.out.println("player"+ node.getPlayer());
//            System.out.println("val:"+val);
//            StatePrinter.printState2D(node.getContext());
//        }
        if(node.getContext().trial().over()){

            return RankUtils.utilities(node.getContext())[node.getPlayer()];
        }
        return network.forward(node.getStateTensor())[1][0];
    }

    /**
     * Method determining rules of selection
     * @param node
     * @return
     */
    public NodeWithWrapper selectionPolicy(NodeWithWrapper node){
        //fdp
//        StatePrinter.logln("selection");
//        StatePrinter.logState2D(node.getContext());
//        StatePrinter.logln("NS: "+node.getN());
        node.setAllFloatP(network);
        FastArrayList<NodeWithWrapper> childnodelist = node.getChildNodeList();

        double bestucbval = -Double.MAX_VALUE;
        NodeWithWrapper bestChildNode = childnodelist.get(0);
        double alphalist[];
        Vec dist = null;
        if(node.getParent()==null&&epsilon!=null){
            //System.out.println("a");
            alphalist = new double[childnodelist.size()];
            for(int i = 0 ; i < childnodelist.size() ; i++){
                alphalist[i]=alpha;
            }
            Vec vec = new DenseVector(alphalist);
            Dirichlet dirichlet = new Dirichlet(vec);
            dist = dirichlet.sample(1,new Random()).get(0);
        }
        for(int i = 0; i < childnodelist.size(); i++) {

            double curUCB;

            if(dist==null){

                curUCB=UCB(node,childnodelist.get(i));
            }else{
                //System.out.println("b");
                curUCB=UCB(node,childnodelist.get(i),(float) dist.get(i));
            }

            //fdp
            //StatePrinter.logln(childnodelist.get(i).getMoveFromParent()+" NSA: "+childnodelist.get(i).getN()+" QSA: "+childnodelist.get(i).getQ()+" PSA: "+childnodelist.get(i).getP()+" ucb: "+curUCB);
            if(curUCB>bestucbval) {
                bestucbval=curUCB;
                bestChildNode = childnodelist.get(i);
            }
        }

        return bestChildNode;
    }

    /**
     * Method to do backpropagation
     * @param node last represented node
     * @param val value of last node
     */
    public void backpropagation(NodeWithWrapper node, double val){
        //fdp
        //StatePrinter.logln("backpropagation:");
        //StatePrinter.logln("backpropagation:");
        //loop the until root
        while(node.getParent()!=null){

            //StatePrinter.logState2D(node.getContext());

            // set up the val to -val move form parent node --> this node worth -val
            val=-val;
            // check if value of curent node available
            if(node.getN()!=null&&node.getQ()!=null){
                // add Q value with val
                //StatePrinter.logState2D(node.getContext());
                //StatePrinter.logln(node.getQ()+"+"+val);
                node.setQ(node.getQ()+val);
                //StatePrinter.logln(node.getQ()+"");

                // if avialable add one visited count
                node.setN(node.getN()+1);

            }else{
                // if not available set the number visited N to 1
                node.setN(1);
                // set the Q value to val
                node.setQ(val);
                //fdp
                //StatePrinter.logState2D(node.getContext());
                //StatePrinter.logln("end"+val);
            }
            // move to parent node
            node = node.getParent();
        }

        // root node needs visit count need to be assigned
        // check root already visited
        if(node.getN()!=null){
            // if unvisited add 1 to root node visit count
            node.setN(node.getN()+1);
        }else{
            // if unvisited set visit  count to 1
            node.setN(1);
        }


    }


    /**
     * Doing one iteration of MCTS
     * @param node node representing the root of tree node
     */
    public void search(NodeWithWrapper node) {
        //get reference of current node computed by algorithm
        NodeWithWrapper cur = node;
        //fdp
        //StatePrinter.logln("Selection:");

        while(true){
            // check if terminal
            if(cur.getContext().trial().over()==true){
                // get rank of the player on current node
                //System.out.println(Arrays.toString(RankUtils.utilities(cur.getContext())));
                //fdp
                //StatePrinter.logln("over:"+val);
                double val = RankUtils.utilities(cur.getContext())[cur.getPlayer()];




                // do backpropagation
                backpropagation(cur,val);
                // end the iteration
                return;
            }

            // check if nonterminal and not all the legal move has been expanded
            if(!cur.getUnexpandedMove().isEmpty()){
                //fdp
                //StatePrinter.logln("expansion:");
               // StatePrinter.logState2D(cur.getContext());

                // do expansion
                expand(cur);
                // store the current newly expanded node
                NodeWithWrapper expandedNode = cur.getChildNodeList().get(cur.getChildNodeList().size()-1);
                // do playout
                double val = playoutPolicy(expandedNode);
                // do backpropagation
                //System.out.println(val);
                backpropagation(expandedNode,val);
                break ;

            }else{
                //fdp
                //StatePrinter.logState2D(cur.getContext());
                // do selection and set the current
                cur = selectionPolicy(cur);
            }

        }


    }

    /**
     * Method used to count UCB value of current NodeWithWrapper object from node to childnode
     * @param node NodeWithWrapper object that where the UCB want to be calculated
     * @param childNode  choosed child object where the UCB want to be calculated
     * @return value of UCB with double data type
     */
    public double UCB(NodeWithWrapper node, NodeWithWrapper childNode){
        // calculate the ucb with Q(v)/N(v')+P(v')*SQRT(log(N(v))/N(v'))
        if(childNode.getQ()!=null) {
            return (childNode.getQ() / childNode.getN()) + (c_uct * childNode.getP() * Math.sqrt((double) node.getN()) / ((double) childNode.getN() + 1));
        }else{
            return Double.MAX_VALUE;
        }
    }

    /**
     * Method used to count UCB value of current NodeWithWrapper object from node to childnode
     * @param node NodeWithWrapper object that where the UCB want to be calculated
     * @param childNode  choosed child object where the UCB want to be calculated
     * @return value of UCB with double data type
     */
    public double UCB(NodeWithWrapper node, NodeWithWrapper childNode,float dirDistValue){
        // calculate the ucb with Q(v)/N(v')+P(v')*SQRT(log(N(v))/N(v'))
        if(childNode.getQ()!=null) {
            float probs = (1-epsilon)*childNode.getP()+epsilon*dirDistValue;
            return (childNode.getQ() / childNode.getN()) + (c_uct * probs * Math.sqrt((double) node.getN()) / ((double) childNode.getN() + 1));
        }else{
            return Double.MAX_VALUE;
        }
    }


    /**
     * Method used to get next move of tree
     * @param node root node tree
     * @return best move that calculated by comparing UCB maximal value
     */
    public Move getBestAction(NodeWithWrapper node) {
        // get all child that contained in the tree
        FastArrayList<NodeWithWrapper> childNodeList = node.getChildNodeList();
        // set temporary maximum value with minimal value
        Integer maxN=-Integer.MAX_VALUE;
        // set variable to store best move

        FastArrayList<Integer> bestAction = new FastArrayList<Integer>();

        // loop throught all the child node
        for(int i = 0 ; i  < childNodeList.size(); i++) {
            // calculate value ucb of : curent node --> childnode of index i
//            System.out.print(childNodeList.get(i).getMoveFromParent()+":");
//            System.out.print(childNodeList.get(i).getN()+",");
//            System.out.println(childNodeList.get(i).getP());
            int curN = childNodeList.get(i).getN();
            // fdp: add ub to ucblist
//            DNpar.add(node.getN());
//            DNcur.add(childNodeList.get(i).getN());
//            DMoveCur.add(childNodeList.get(i).getMoveFromParent());
//            DP.add(childNodeList.get(i).getP());
//            DQ.add(childNodeList.get(i).getQ());
            //DUCB.add(curUCB);
            // check whether current node --> childnode of index i UCB value is maximum
            if(curN>maxN) {
                // change temporary maximum UCB value
                maxN=curN;
                bestAction.clear();
                bestAction.add(i);
            }else if(curN==maxN){
                bestAction.add(i);
            }
        }
        int bestidx =  bestAction.get(ThreadLocalRandom.current().nextInt(bestAction.size()));
        //System.out.println();
        //System.out.println();

        //fdp
//        StatePrinter.logln("move"+DMoveCur);
//        StatePrinter.logln("npar"+DNpar);
//        StatePrinter.logln("ncur"+DNcur);
//        StatePrinter.logln("p"+DP);
//        StatePrinter.logln("q"+DQ);
//        StatePrinter.logln("ucb"+DUCB);





        //return the result
        return childNodeList.get(bestidx).getMoveFromParent();
    }


    public NodeWithWrapper getBestNode(NodeWithWrapper node) {
        // get all child that contained in the tree
        FastArrayList<NodeWithWrapper> childNodeList = node.getChildNodeList();
        // set temporary maximum value with minimal value
        Integer maxN=-Integer.MAX_VALUE;
        // set variable to store best move

        FastArrayList<Integer> bestAction = new FastArrayList<Integer>();

        //fdp
        //StatePrinter.printState2D(node.getContext());
        // loop throught all the child node
        for(int i = 0 ; i  < childNodeList.size(); i++) {
            int curN = childNodeList.get(i).getN();
            //fdp
            //System.out.println(childNodeList.get(i).getMoveFromParent()+" "+childNodeList.get(i).getN()+" "+childNodeList.get(i).getP());
            if(curN>maxN) {
                // change temporary maximum UCB value
                maxN=curN;
                bestAction.clear();
                bestAction.add(i);
            }else if(curN==maxN){
                bestAction.add(i);
            }
        }
        int bestidx =  bestAction.get(ThreadLocalRandom.current().nextInt(bestAction.size()));

        return childNodeList.get(bestidx);
    }






    public void forceExpand(NodeWithWrapper parentnode,Move move){
        MCTS.MoveKey mkey = new MCTS.MoveKey(move,0);
//        System.out.println("searched"+mkey);
//        for(MCTS.MoveKey m: parentnode.getUnexpandedMove()){
//            System.out.println(m);
//            System.out.println(m.equals(mkey));
//        }
        if(parentnode.getUnexpandedMove().contains(mkey)){
            // create new context
            Context context = new Context(parentnode.getContext());
            // remove next move
            parentnode.getUnexpandedMove().remove(mkey);
            // do move to the context
            context.game().apply(context,mkey.move);
            // make new node after applying the move to the context
            NodeWithWrapper newnode = new NodeWithWrapper(parentnode,context,nextPlayer(parentnode.getPlayer()),mkey.move);
            // add node to the tree
            parentnode.addChild(newnode);
        }else{
            System.out.println("forced expand failed");
        }
    }


}