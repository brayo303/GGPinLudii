
import DataStructure.TrainIter;

import game.Game;
import main.collections.FastArrayList;

import other.AI;
import other.RankUtils;
import other.context.Context;
import other.move.Move;


import java.util.concurrent.ThreadLocalRandom;


public class MyAgent extends AI
{

	private  int num_it = 10000;
	private Node node;
	
	
	protected int player = -1;
	
	
	protected Move lastReturnedMove = null;
	
	
	
	public MyAgent()
	{
		friendlyName = "UCT";
		
	}

	public MyAgent(int it){
		friendlyName = "UCT"+it;
		num_it=it;
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
		
		//gameWrapper = new CustomGameWrapper(game);
		//stateWrapper = new CustomStateWrapper(gameWrapper,context);
		MCTS mcts = new MCTS();
		node = new Node(null,context,player,null);
		for(int i = 0 ; i < num_it ; i++) {
			mcts.search(node);
		}
		//StatePrinter.logln(getActionProbability(game)+"");
		return mcts.getBestAction(node);
		
	}





	
	@Override
	public void initAI(final Game game, final int playerID)
	{
		this.player = playerID;
		lastReturnedMove = null;
		
		/*System.out.println(Arrays.toString(gameWrapper.stateTensorChannelNames()));
		System.out.println("moveTensorShape="+Arrays.toString(gameWrapper.moveTensorsShape()));
		System.out.println("stateTensorShape="+Arrays.toString(gameWrapper.stateTensorsShape()));*/
	}
	
	//-------------------------------------------------------------------------

}


class MCTS {
	
	private final double c_uct=1;
	
	
	private int playerNum = 2;
	//if leaf jalankan expand
	
	public int nextPlayer(int player) {
		return (((player-1)+1)%playerNum)+1;
	}
	
	public void expand(Node node) {
		// buat context baru
		Context context = new Context(node.getContext());
		// lakukan move pada context
		//System.out.println(node.getLegalMove().size());
		context.game().apply(context, node.getLegalMove().get(node.getChildNodeList().size()));
		//Move Ns
		Node newnode = new Node(node,context,nextPlayer(node.getPlayer()),node.getLegalMove().get(node.getChildNodeList().size()));
		// tambahkan ke tree
		node.addChild(newnode);


	}
	
	public double playoutPolicy(Node node) {
		Context contextEnd = new Context(node.getContext());

		FastArrayList<Move> nextMoves;

		while(!contextEnd.trial().over()) {
			nextMoves = new FastArrayList<Move>(contextEnd.game().moves(contextEnd).moves());
			contextEnd.game().apply(contextEnd, nextMoves.get(ThreadLocalRandom.current().nextInt(nextMoves.size())));

		}
//		System.out.println("end:");
//		StatePrinter.printState2D(contextEnd);
		return RankUtils.utilities(contextEnd)[node.getPlayer()];
	}

	public Node selectionPolicy(Node node){
		FastArrayList<Node> childnodelist = node.getChildNodeList();
		double bestucbval = -Double.MAX_VALUE;
		Node bestChildNode = childnodelist.get(0);
		for(int i = 0; i < childnodelist.size(); i++) {
			double curUCB = UCB(node,childnodelist.get(i));
			if(curUCB>bestucbval) {
				bestucbval=curUCB;
				bestChildNode = childnodelist.get(i);
			}
		}

		return bestChildNode;
	}

	public void backpropagation(Node node, double val){
		while(node.getParent()!=null){
			val=-val;
			if(node.getN()!=null&&node.getQ()!=null){
				node.setN(node.getN()+1);
				node.setQ(node.getQ()+val);
			}else{
				node.setN(1);
				node.setQ(val);

			}
			//System.out.println("val:"+Q.get(node));

			node = node.getParent();
		}

		// for parent
		if(node.getN()!=null){
			node.setN(node.getN()+1);
		}else{
			node.setN(1);
		}


	}
	
	
	public void search(Node node) {


		Node cur = node;

		while(true){
			//terminal
			if(cur.getContext().trial().over()==true){
				double val = RankUtils.utilities(cur.getContext())[cur.getPlayer()];
				//backpropagation
				backpropagation(cur,val);
				return;
			}

			//nonterminal
			if(cur.getChildNodeList().size()<cur.getLegalMove().size()){
				//expansion
				expand(cur);
				//expanded node
				Node expandedNode = cur.getChildNodeList().get(cur.getChildNodeList().size()-1);
				//playout
				double val = playoutPolicy(expandedNode);
				//backpropagation
				backpropagation(expandedNode,val);
				break ;

			}else{
				//selection
				cur = selectionPolicy(cur);
			}

		}
		
	}
	
	public double UCB(Node node, Node childNode){
		return (childNode.getQ()/(double)childNode.getN())+c_uct*Math.sqrt(Math.log((double)node.getN())/(double)childNode.getN());
	}
	
	
	public Move getBestAction(Node node) {
		FastArrayList<Node> childNodeList = node.getChildNodeList();
		int maxN=0;
		Move choosedMove= null;

		for(int i = 0 ; i  < childNodeList.size(); i++) {

			if(childNodeList.get(i).getN()>maxN) {
				maxN = childNodeList.get(i).getN();
				choosedMove=childNodeList.get(i).getMoveFromParent();

			}

		}

		//System.out.println(choosedMove);
		return choosedMove;
	}


}











class Node {
	//parent node
	private Node parent;
	private FastArrayList<Move> legalMoves;
	private Context context;
	private int currentPlayer;
	private FastArrayList<Node> childNodeList;
	private Move moveFromParent;
	private Integer N;
	private Double Q;






	public Node(Node parent, Context context,int currentPlayer, Move moveFromParent) {
		this.parent=parent;
		this.context=context;
		this.currentPlayer=currentPlayer;
		this.childNodeList=new FastArrayList<Node>();
		this.legalMoves= new FastArrayList<Move>();
		this.moveFromParent = moveFromParent;
		this.generateMove();
		this.N = null;
		this.Q = null;

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
		legalMoves = new FastArrayList<Move>(game.moves(context).moves());

	}



	public FastArrayList<Move> getLegalMove(){
		return legalMoves;
	}

	public FastArrayList<Node> getChildNodeList(){
		return childNodeList;
	}


	public Node getParent() {
		return parent;
	}

	public void addChild(Node childNode) {
		childNodeList.add(childNode);
	}



	public Context getContext() {
		return context;
	}

	public Move getMoveFromParent() {
		return moveFromParent;
	}



	public int getPlayer() {
		return currentPlayer;
	}
}


 



