import java.awt.Color;
import java.util.List;
import java.util.Random;

import burlap.behavior.singleagent.*;
import burlap.domain.singleagent.gridworld.*;
import burlap.oomdp.core.*;
import burlap.oomdp.singleagent.*;
import burlap.oomdp.singleagent.common.*;
import burlap.behavior.statehashing.DiscreteStateHashFactory;
import burlap.behavior.singleagent.learning.*;
import burlap.behavior.singleagent.learning.tdmethods.*;
import burlap.behavior.singleagent.planning.*;
import burlap.behavior.singleagent.planning.commonpolicies.GreedyQPolicy;
import burlap.behavior.singleagent.planning.deterministic.*;
import burlap.behavior.singleagent.planning.deterministic.informed.Heuristic;
import burlap.behavior.singleagent.planning.deterministic.informed.astar.AStar;
import burlap.behavior.singleagent.planning.deterministic.uninformed.bfs.BFS;
import burlap.behavior.singleagent.planning.deterministic.uninformed.dfs.DFS;
import burlap.behavior.singleagent.planning.stochastic.policyiteration.PolicyIteration;
import burlap.behavior.singleagent.planning.stochastic.valueiteration.ValueIteration;
import burlap.oomdp.visualizer.Visualizer;
import burlap.oomdp.auxiliary.StateGenerator;
import burlap.oomdp.auxiliary.StateParser;
import burlap.oomdp.auxiliary.common.ConstantStateGenerator;
import burlap.behavior.singleagent.EpisodeSequenceVisualizer;
import burlap.behavior.singleagent.auxiliary.StateReachability;
import burlap.behavior.singleagent.auxiliary.performance.LearningAlgorithmExperimenter;
import burlap.behavior.singleagent.auxiliary.performance.PerformanceMetric;
import burlap.behavior.singleagent.auxiliary.performance.TrialMode;
import burlap.behavior.singleagent.auxiliary.valuefunctionvis.ValueFunctionVisualizerGUI;
import burlap.behavior.singleagent.auxiliary.valuefunctionvis.common.*;
import burlap.behavior.singleagent.auxiliary.valuefunctionvis.common.PolicyGlyphPainter2D.PolicyGlyphRenderStyle;
import burlap.oomdp.singleagent.common.VisualActionObserver;

public class MDP1 {
	GridWorldDomain 			gwdg;
	Domain						domain;
	StateParser 				sp;
	RewardFunction 				rf;
	TerminalFunction			tf;
	StateConditionTest			goalCondition;
	State 						initialState;
	DiscreteStateHashFactory	hashingFactory;
	
	int							width=15;
	int							height=15;
	static int 					seed=1;
	public static void main(String[] args) {
		
		
		MDP1 maze = new MDP1();
		String outputPath = "output/"; 
		
		
		
		long startTime=System.currentTimeMillis();
		maze.ValueIterationExample(outputPath);
		long duration=System.currentTimeMillis()-startTime;
		System.out.printf("Duration= %4.3f s", duration/1000.0);
		//run the visualizer (only use if you don't use the experiment plotter example)
		maze.visualize(outputPath);
		
		startTime=System.currentTimeMillis();
		maze.PolicyIterationExample(outputPath);
		duration=System.currentTimeMillis()-startTime;
		System.out.printf("Duration= %4.3f s", duration/1000.0);
		//run the visualizer (only use if you don't use the experiment plotter example)
		maze.visualize(outputPath);
		
	}
	
	
	public MDP1(){
	
		//create the domain
		gwdg = new GridWorldDomain(width, height);
		setWalls(); 
		gwdg.setProbSucceedTransitionDynamics(0.8); //stochastic transitions with 0.8 success rate
		domain = gwdg.generateDomain();
		
			
//		/*This part sets up the reward function to a standard value at all locations then sets two specific locations to different values one positive and one negative.*/
		rf = new GridWorldRewardFunction(width, height, -0.04);
		((GridWorldRewardFunction) rf).setReward(width-1,height-1,2);
		((GridWorldRewardFunction) rf).setReward(width/2,height/2,-2);
//		((GridWorldRewardFunction) rf).setReward(width/4,height/4*3,-2);
//		((GridWorldRewardFunction) rf).setReward(width/4*3,height/4,-2);
		
//		/*This part sets up the transition function to have an intial terminal state, and then adds a second terminal state.*/	
//		tf = new GridWorldTerminalFunction(3,2);
//		((GridWorldTerminalFunction)tf).markAsTerminalPosition(3, 1);
		
		
		
		//create the state parser
		sp = new GridWorldStateParser(domain); 
		
		//define the task
		//rf = new UniformCostRF(); 
		tf = new SinglePFTF(domain.getPropFunction(GridWorldDomain.PFATLOCATION)); 
		goalCondition = new TFGoalCondition(tf);
		
		//set up the initial state of the task
		//initialState = GridWorldDomain.getOneAgentOneLocationState(domain);
		initialState = GridWorldDomain.getOneAgentNLocationState(domain,2);
		GridWorldDomain.setAgent(initialState, 0, 0);
		GridWorldDomain.setLocation(initialState, 0, width-1,height-1);
		GridWorldDomain.setLocation(initialState, 1, width/2,height/2);
		
		//set up the state hashing system
		hashingFactory = new DiscreteStateHashFactory();
		hashingFactory.setAttributesForClass(GridWorldDomain.CLASSAGENT, 
				domain.getObjectClass(GridWorldDomain.CLASSAGENT).attributeList); 
				
				
		//add visual observer
		VisualActionObserver observer = new VisualActionObserver(domain, 
			GridWorldVisualizer.getVisualizer(domain, gwdg.getMap()));
		((SADomain)this.domain).setActionObserverForAllAction(observer);
		observer.initGUI();		
		
			
	}
	
	private static int[][] RandomArray(int m, int n) {
	    int[][] randomMatrix = new int [m][n];

	    Random rand = new Random(); 
	    //rand.setSeed(seed);   // fix a random map for repeats
	    for (int i = 0; i < m; i++) {     
	        for (int j = 0; j < n; j++) {
	        	
	            Integer r = rand.nextInt(20); 
	            if (r>2) {randomMatrix[i][j] = 0;}
	            else {randomMatrix[i][j] = 1;}
	        }

	    }

	    return randomMatrix;
	}

	public void setWalls(){
		gwdg.setMap(RandomArray(width,height));
	}
	
	public void visualize(String outputPath){
		Visualizer v = GridWorldVisualizer.getVisualizer(domain, gwdg.getMap());
		EpisodeSequenceVisualizer evis = new EpisodeSequenceVisualizer(v, 
								domain, sp, outputPath);
	}
	
	
	
	public void ValueIterationExample(String outputPath){
		
		if(!outputPath.endsWith("/")){
			outputPath = outputPath + "/";
		}
		
		
		OOMDPPlanner planner = new ValueIteration(domain, rf, tf, 0.8, hashingFactory,
								0.001, 100);
		planner.planFromState(initialState);
		System.out.println("mark");
		//create a Q-greedy policy from the planner
		Policy p = new GreedyQPolicy((QComputablePlanner)planner);
		
		//record the plan results to a file
		p.evaluateBehavior(initialState, rf, tf).writeToFile(outputPath + "planResult", sp);
		
		//visualize the value function and policy
		this.valueFunctionVisualize((QComputablePlanner)planner, p);
		
	}
	
	public void PolicyIterationExample(String outputPath){
		
		if(!outputPath.endsWith("/")){
			outputPath = outputPath + "/";
		}
		
		
		OOMDPPlanner planner = new PolicyIteration(domain, rf, tf, 0.8, hashingFactory,
								0.001, 0.001, 100, 100);
		planner.planFromState(initialState);

		//create a Q-greedy policy from the planner
		Policy p = new GreedyQPolicy((QComputablePlanner)planner);
		
		//record the plan results to a file
		p.evaluateBehavior(initialState, rf, tf).writeToFile(outputPath + "planResult", sp);
		
		//visualize the value function and policy
		this.valueFunctionVisualize((QComputablePlanner)planner, p);
		
	}
	
	public void QLearningExample(String outputPath){
		
		if(!outputPath.endsWith("/")){
			outputPath = outputPath + "/";
		}
		
		//discount= 0.99; initialQ=0.0; learning rate=0.9
		LearningAgent agent = new QLearning(domain, rf, tf, 0.99, hashingFactory, 0., 0.9);
		
		//run learning for 100 episodes
		for(int i = 0; i < 100; i++){
			EpisodeAnalysis ea = agent.runLearningEpisodeFrom(initialState);
			ea.writeToFile(String.format("%se%03d", outputPath, i), sp); 
			System.out.println(i + ": " + ea.numTimeSteps());
		}
		
	}
	
	public void valueFunctionVisualize(QComputablePlanner planner, Policy p){
		List <State> allStates = StateReachability.getReachableStates(initialState, 
			(SADomain)domain, hashingFactory);
		LandmarkColorBlendInterpolation rb = new LandmarkColorBlendInterpolation();
		rb.addNextLandMark(0., Color.RED);
		rb.addNextLandMark(1., Color.BLUE);
		
		StateValuePainter2D svp = new StateValuePainter2D(rb);
		svp.setXYAttByObjectClass(GridWorldDomain.CLASSAGENT, GridWorldDomain.ATTX, 
			GridWorldDomain.CLASSAGENT, GridWorldDomain.ATTY);
		
		PolicyGlyphPainter2D spp = new PolicyGlyphPainter2D();
		spp.setXYAttByObjectClass(GridWorldDomain.CLASSAGENT, GridWorldDomain.ATTX, 
			GridWorldDomain.CLASSAGENT, GridWorldDomain.ATTY);
		spp.setActionNameGlyphPainter(GridWorldDomain.ACTIONNORTH, new ArrowActionGlyph(0));
		spp.setActionNameGlyphPainter(GridWorldDomain.ACTIONSOUTH, new ArrowActionGlyph(1));
		spp.setActionNameGlyphPainter(GridWorldDomain.ACTIONEAST, new ArrowActionGlyph(2));
		spp.setActionNameGlyphPainter(GridWorldDomain.ACTIONWEST, new ArrowActionGlyph(3));
		spp.setRenderStyle(PolicyGlyphRenderStyle.DISTSCALED);
		
		ValueFunctionVisualizerGUI gui = new ValueFunctionVisualizerGUI(allStates, svp, planner);
		gui.setSpp(spp);
		gui.setPolicy(p);
		gui.setBgColor(Color.GRAY);
		gui.initGUI();
	}

}
