package DataStructure;

import Block.NeuralNetworkManager;
import TestUtil.Debug;
import TestUtil.StatePrinter;
import customWrapper.CustomGameWrapper;
import customWrapper.CustomStateWrapper;
import game.Game;
import main.collections.FastArrayList;
import other.context.Context;
import other.move.Move;
import search.mcts.MCTS;

import java.util.*;

public class NodeWithWrapper {
    //parent node
    private NodeWithWrapper parent;

    private Context context;
    private int currentPlayer;
    private FastArrayList<NodeWithWrapper> childNodeList;
    private Set<MCTS.MoveKey> unexpandedMove;
    private Map<MCTS.MoveKey,NodeWithWrapper> expandedMoveMap;
    private Move moveFromParent;
    private Integer N;
    private Double Q;
    private Float P;
    float [][][] stateTensor;

    CustomGameWrapper gameWrapper;

    private boolean allProbabilitySet;



    public NodeWithWrapper(NodeWithWrapper parent, Context context, int currentPlayer, Move moveFromParent) {
        this.parent=parent;
        this.context=context;
        this.currentPlayer=currentPlayer;
        this.childNodeList=new FastArrayList<NodeWithWrapper>();
        this.moveFromParent = moveFromParent;
        this.expandedMoveMap = new HashMap<>();
        this.N = null;
        this.Q = null;
        this.allProbabilitySet=false;
        gameWrapper= new CustomGameWrapper(context.game());
        CustomStateWrapper stateWrapper = new CustomStateWrapper(gameWrapper,context);
        stateTensor = stateWrapper.toTensor();
//        if(parent==null) {
//            for(int i = 0 ; i <stateTensor[1].length ; i++){
//                for(int j = 0 ; j < stateTensor[1][i].length;j++){
//                    System.out.print(stateTensor[1][i][j]+ " ");
//                }
//                 System.out.println();
//            }
//
//        }

    }

    public void setP(Float p) {
        P = p;
    }

    public float[][][] getStateTensor() {
        return stateTensor;
    }

    public Float getP() {
        return P;
    }

    public Integer getN() {
        return N;
    }

    public void setN(Integer n) {
        N = n;
    }

    public Double getQ() {
        return Q;
    }

    public void setQ(Double q) {
        Q = q;
    }

    private void generateMove() {
        final Game game = context.game();
        FastArrayList <Move> legalMoves = new FastArrayList<>(game.moves(context).moves());
        this.unexpandedMove = new HashSet<>();
        for(int i = 0 ; i < legalMoves.size() ; i++){
            unexpandedMove.add(new MCTS.MoveKey(legalMoves.get(i),0));
        }
    }

    public Set<MCTS.MoveKey> getUnexpandedMove() {
        if(unexpandedMove==null){
            generateMove();
        }
        return unexpandedMove;
    }



    public void setAllFloatP(NeuralNetworkManager net){
        if(!allProbabilitySet) {


            float nnet[] = net.forward(stateTensor)[0];

            if(parent==null) {
//                for(int i = 0 ; i < stateTensor[1].length; i++){
//                    String temp="";
//                    for(int j = 0 ; j < stateTensor[1][i].length; j++){
//                        temp+=stateTensor[1][i][j]+" ";
//                    }
//                    StatePrinter.logln(temp);
//                }
//                StatePrinter.logln("===");
                Debug.println(Arrays.toString(nnet));
            }
            float sum = 0;

            for (int i = 0; i < childNodeList.size(); i++) {
                int index = gameWrapper.moveToInt(childNodeList.get(i).getMoveFromParent());
                sum += nnet[index];
                //System.out.print(nnet[index]+" ");
                childNodeList.get(i).setP(nnet[index]);
            }
            //System.out.println();

//          if(parent==null){
////                //net.debugWeight();
////                //System.out.println(Arrays.toString(stateTensor));
////                //System.out.println(Arrays.toString(nnet));
//                for (int i = 0; i < childNodeList.size(); i++) {
////                    int index = gameWrapper.moveToInt(childNodeList.get(i).getMoveFromParent());
////                    System.out.print(childNodeList.get(i).getMoveFromParent()+":");
////                    System.out.print(index+",");
////                    System.out.print(nnet[gameWrapper.moveToInt(childNodeList.get(i).getMoveFromParent())]);
//                    StatePrinter.logln(childNodeList.get(i).getMoveFromParent()+":"+gameWrapper.moveToInt(childNodeList.get(i).getMoveFromParent()));
//                    StatePrinter.logln(nnet[gameWrapper.moveToInt(childNodeList.get(i).getMoveFromParent())]+"");
//                }
////               // System.out.println("");
//        }

            if (sum <= 0) {
                StatePrinter.filename = "warning";
                StatePrinter.logln("Miss All Predicted prob (all 0)");
                System.out.println("Miss All Predicted prob (all 0)");
                for (int i = 0; i < childNodeList.size(); i++) {
                    childNodeList.get(i).setP(1.0f);
                }
            } else {
                for (int i = 0; i < childNodeList.size(); i++) {
                    childNodeList.get(i).setP(childNodeList.get(i).getP() / sum);
                }
            }
            allProbabilitySet=true;
        }

    }





    public FastArrayList<NodeWithWrapper> getChildNodeList(){
        return childNodeList;
    }


    public NodeWithWrapper getParent() {
        return parent;
    }

    public void addChild(NodeWithWrapper childNode) {

        childNodeList.add(childNode);
        expandedMoveMap.put(new MCTS.MoveKey(childNode.getMoveFromParent(),0),childNode);
    }



    public Context getContext() {
        return context;
    }

    public Move getMoveFromParent() {
        return moveFromParent;
    }

    public NodeWithWrapper searchChildNode(Move move){
        return expandedMoveMap.get(new MCTS.MoveKey(move,0));
    }


    public int getPlayer() {
        return currentPlayer;
    }
}
