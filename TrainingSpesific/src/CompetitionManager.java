import Block.NeuralNetworkManager;
import DataStructure.NodeWithWrapper;
import game.Game;
import other.AI;
import other.context.Context;
import other.move.Move;
import other.trial.Trial;

import java.util.ArrayList;
import java.util.List;

public class CompetitionManager {

    private Game game;

    public CompetitionManager(Game game){
        this.game = game;
    }


    @Deprecated
    public int[] compete(NeuralNetworkManager prev , NeuralNetworkManager cur , int numIt, String... warning) {

        //0 network prev win;
        //1 network cur win
        int netwin[] = new int[2];
        MCTSDeepLearning mcts[] = new MCTSDeepLearning[2];
        NodeWithWrapper parentnode[] = new NodeWithWrapper[2];
        NodeWithWrapper curnode[] = new NodeWithWrapper[2];
        Context firstcontext = new Context(game, new Trial(game));
        game.start(firstcontext);
        parentnode[0] = new NodeWithWrapper(null,firstcontext , 1, null);
        mcts[0] = new MCTSDeepLearning(prev);
        mcts[1] = new MCTSDeepLearning(cur);



        //compete one time;
        for (int i = 0; i < numIt; i++) {
            curnode[0] = parentnode[0];
            curnode[1] = parentnode[1];
            Context curContext = new Context(game, new Trial(game));
            game.start(curContext);

            while (!curContext.trial().over()) {
                int mover = curContext.state().mover() - 1;
                //System.out.println(mover);
                for (int j = 0; j < Configuration.mcts_it_eval; j++) {
                    mcts[mover].search(curnode[mover]);
                }
                NodeWithWrapper bestNode = mcts[mover].getBestNode(curnode[mover]);
                //System.out.println(bestNode.getChildNodeList());
                game.apply(curContext, bestNode.getMoveFromParent());
                int nextPlayer = curContext.state().mover() - 1;
                if (curnode[nextPlayer] == null) {

                    curnode[nextPlayer] = new NodeWithWrapper(null, new Context(curContext), nextPlayer + 1, null);
                } else {

                    NodeWithWrapper nextNode = curnode[nextPlayer].searchChildNode(bestNode.getMoveFromParent());

                    if (nextNode == null) {
                        mcts[nextPlayer].forceExpand(curnode[nextPlayer], bestNode.getMoveFromParent());
                        nextNode = curnode[nextPlayer].searchChildNode(bestNode.getMoveFromParent());
                    }
                    curnode[nextPlayer] = nextNode;

                }
                //debug
//                StatePrinter.filename="movecheck.txt";
//                StatePrinter.logState2D(curnode[mover].getContext());
//                for(int j = 0 ; j < curnode[mover].getChildNodeList().size() ;  j++){
//                    StatePrinter.logln(curnode[mover].getChildNodeList().get(j).getMoveFromParent()+":");
//                    StatePrinter.logln(curnode[mover].getChildNodeList().get(j).getN()+"");
//                }
//                StatePrinter.logln("======");
//                StatePrinter.logln("bestMove"+bestNode.getMoveFromParent());
//                StatePrinter.logln("======");

                curnode[mover] = bestNode;




            }
            //StatePrinter.logln("****");

            if (curContext.trial().status().winner() == 1) {
                netwin[0] += 1;
                try {
                    System.out.println(warning[0]);
                }catch (Exception e){

                }
            } else if (curContext.trial().status().winner() == 2) {
                netwin[1] += 1;
                try{
                    System.out.println(warning[1]);
                }catch (Exception e){

                }
            } else{
                try{
                    System.out.println(warning[2]);
                }catch (Exception e){

                }
            }


        }

        //System.out.println(curnode[0].getN());
        return netwin;
    }


    public int[] compete(TrainAgent prev , TrainAgent cur , int numIt, String... warning){
        int netwin[] = new int[2];
        final List<AI> agents = new ArrayList<>();
        agents.add(null);
        agents.add(prev);
        agents.add(cur);
        Trial trial = new Trial(game);
        Context context = new Context(game, trial);
        game.start(context);
        NodeWithWrapper node = new NodeWithWrapper(null, context, 1, null);
        NodeWithWrapper nodeb = new NodeWithWrapper(null, context, 1, null);
        ((TrainAgent)agents.get(1)).initAI(game, 1,node,true);
        ((TrainAgent)agents.get(2)).initAI(game, 2,nodeb,false);
        for(int i  =0 ; i < numIt ; i++) {

            trial = new Trial(game);
            context = new Context(game, trial);
            game.start(context);



            int mover=0;
            while (!context.trial().over()) {
                mover = context.state().mover();

                TrainAgent agent = (TrainAgent) agents.get(mover);

                //System.out.println("sizeawal"+agent.getRoot().getUnexpandedMove().size());
                final Move move = agent.selectAction(game, context, 0.2, -1, -1);
                //fdp :: geting child node move

//                if(agent.getNodeNow().getParent()!=null){
//                    System.out.println("parent:"+agent.getNodeNow().getParent().getPlayer());
//                }else{
//                    System.out.println("noparent");
//                }
//                for(int it = 0 ; it < agent.getRoot().getChildNodeList().size() ; it++){
//                    System.out.print(agent.getRoot().getChildNodeList().get(it).getN()+" ");
//                }
//                System.out.println();
//                    StatePrinter.printState2D(agent.getNodeNow().getContext());
//               System.out.println("root:"+agent.getRoot().getN());
//               for(int j = 0 ; j < prev.getRoot().getChildNodeList().size() ; j++){
//                   System.out.println("child:"+prev.getRoot().getChildNodeList().get(j).getN());
//               }
//                System.out.println(mover);
//                StatePrinter.printlnpiece(agent.getNodeNow().getContext());
//                System.out.println(node.getN());
//                System.out.println(nodeb.getN());

                //agent.getNetwork();
                game.apply(context, move);




            }
            //System.out.println("+++");
            //StatePrinter.printState2D(context);
            //fdp
            //System.out.println("end");
            if (context.trial().status().winner() == 1) {
                netwin[0] += 1;
                try {
                    System.out.println(warning[0]);
                }catch (Exception e){

                }
            } else if (context.trial().status().winner() == 2) {
                netwin[1] += 1;
                try{
                    System.out.println(warning[1]);
                }catch (Exception e){

                }
            } else{
                try{
                    System.out.println(warning[2]);
                }catch (Exception e){

                }
            }



            prev.resetFirstPlay();
            prev.resetNode();
            cur.resetNode();
            if(mover==1){
                cur.forceMCTS();
            }

        }

        return netwin;
    }


}
