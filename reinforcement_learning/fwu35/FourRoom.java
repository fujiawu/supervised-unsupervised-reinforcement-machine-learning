import java.awt.Color;
import java.util.List;

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
import burlap.behavior.singleagent.planning.commonpolicies.EpsilonGreedy;
import burlap.behavior.singleagent.planning.deterministic.*;
import burlap.behavior.singleagent.planning.deterministic.informed.Heuristic;
import burlap.behavior.singleagent.planning.deterministic.informed.astar.AStar;
import burlap.behavior.singleagent.planning.deterministic.uninformed.bfs.BFS;
import burlap.behavior.singleagent.planning.deterministic.uninformed.dfs.DFS;
import burlap.behavior.singleagent.planning.stochastic.valueiteration.ValueIteration;
import burlap.behavior.singleagent.planning.stochastic.policyiteration.PolicyIteration;
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

public class FourRoom {

	GridWorldDomain 			gwdg;
	Domain			        	domain;
	StateParser 				sp;
	RewardFunction 				rf;
	TerminalFunction			tf;
	StateConditionTest			goalCondition;
	State 					initialState;
	DiscreteStateHashFactory        	hashingFactory;

        double discount = 0.99;
        double maxDelta = 0.001;
        int maxIterations = 1000000;
        double learn_rate = 0.9;
        double initialQ = 0;
        double lambda = 1.0;
        double epsilon = 0.1;
 
	public static void main(String[] args) {
                if (args.length != 2) {
                        System.out.println("Need to provide an algorithm choice and size of state");
                        return;
                }
                int learn_algorithm;
                if (args[0].equals("value")) {
                        System.out.println("Performing Value Iteration");
                        learn_algorithm = 0;
                } else if (args[0].equals("policy")) {
                        System.out.println("Performing Policy Iteration");
                        learn_algorithm = 1;
                } else if (args[0].equals("qlearn")) {
                        System.out.println("Performing Q Learning");
                        learn_algorithm = 2;
                } else if (args[0].equals("experiment")) {
                        System.out.println("Performing Experiment and Plotter");
                        learn_algorithm = 3;
                }
                  else {
                        System.out.println("Don't understand this algorithm choice");
                        return;
                }
	
                int N = Integer.parseInt(args[1]);
                if (N < 7) {
                        System.out.println("N has to be larger than 6");
                        return;
                }
		FourRoom mdp = new FourRoom(N);
		String outputPath = "output/";
                       
                switch(learn_algorithm) {
                  case 0: mdp.ValueIteration(outputPath);
                        break;
                  case 1: mdp.PolicyIteration(outputPath);
                        break;
                  case 2: mdp.QLearning(outputPath);
                        break;
                  case 3: mdp.experimenterAndPlotter();
                        break;
                  default: break;
               }

		mdp.visualize(outputPath);
		
                return;
	}

        protected double p = 0.8;

        protected double[][] transition = new double [][]{
           {p,  0.0, (1-p)/2, (1-p)/2},
           {0.0, p,  (1-p)/2, (1-p)/2},
           {(1-p)/2, (1-p)/2, p, 0.0},
           {(1-p)/2, (1-p)/2, 0.0, p},
        };
        
        
        public void setFourRooms(GridWorldDomain gwdg) {
                int height = gwdg.getHeight();
                int width = gwdg.getWidth();
                int mid_height = (height-1)/2;
                int mid_width = (width-1)/2;
                int[][] map = new int[width][height];
                for (int i=0; i < width; i++) {
                    for (int j=0; j<height; j++) {
                        if (i==mid_width || j==mid_height) {
                           map[i][j] = 1;
                        } else {
                           map[i][j] = 0;
                        }
                    }
                }
                map[mid_width][(height-1+mid_height)/2] = 0;
                map[mid_width][(mid_height+0)/2] = 0;
                map[(mid_width+width-1)/2][mid_height] = 0;
                map[(mid_width+0)/2][mid_height] = 0;
                gwdg.setMap(map);
                

        }
 
	public FourRoom(int N){
	
		//create the domain
		gwdg = new GridWorldDomain(N, N);
		setFourRooms(gwdg);
                //gwdg.setTransitionDynamics(transition);
                gwdg.setProbSucceedTransitionDynamics(0.8);
		domain = gwdg.generateDomain();
		
		//create the state parser
		sp = new GridWorldStateParser(domain); 
		
		//define the reward function
                rf = new GridWorldRewardFunction(domain, -1);
                ((GridWorldRewardFunction)rf).setReward(N-1, 0, 5);
                ((GridWorldRewardFunction)rf).setReward(0, N-1, 5);
                ((GridWorldRewardFunction)rf).setReward(N-1, N-1, 10);

  
                //define the termination function
		tf = new SinglePFTF(domain.getPropFunction(GridWorldDomain.PFATLOCATION)); 
		goalCondition = new TFGoalCondition(tf);
		
		//set up the initial state of the task
		initialState = GridWorldDomain.getOneAgentNLocationState(domain, 3);
		GridWorldDomain.setAgent(initialState, 0, 0);
		GridWorldDomain.setLocation(initialState, 0, 0, N-1, 0);
		GridWorldDomain.setLocation(initialState, 1, N-1, N-1, 0);
		GridWorldDomain.setLocation(initialState, 2, N-1, 0, 0);

		
		//set up the state hashing system
		hashingFactory = new DiscreteStateHashFactory();
		hashingFactory.setAttributesForClass(GridWorldDomain.CLASSAGENT, 
				domain.getObjectClass(GridWorldDomain.CLASSAGENT).attributeList); 
				
		//add visual observer
		VisualActionObserver observer = new VisualActionObserver(domain, 
			GridWorldVisualizer.getVisualizer(gwdg.getMap()));
		((SADomain)this.domain).setActionObserverForAllAction(observer);
		observer.initGUI();		
	
	}
	
	public FourRoom() {
               this(10);
        }


	public void visualize(String outputPath){
		Visualizer v = GridWorldVisualizer.getVisualizer(gwdg.getMap());
		EpisodeSequenceVisualizer evis = new EpisodeSequenceVisualizer(v, 
								domain, sp, outputPath);
	}
	
	
	public void ValueIteration(String outputPath){
		
		if(!outputPath.endsWith("/")){
			outputPath = outputPath + "/";
		}
	
                long startTime = System.nanoTime();
	
		OOMDPPlanner planner = new ValueIteration(domain, rf, tf, discount, hashingFactory,
								maxDelta, maxIterations);

		planner.planFromState(initialState);
                long duration = System.nanoTime() - startTime;
                System.out.println("Running time:"+ (double)duration/1000000000+"s");

		//create a Q-greedy policy from the planner
		Policy p = new GreedyQPolicy((QComputablePlanner)planner);
		
		//record the plan results to a file
	        p.evaluateBehavior(initialState, rf, tf).writeToFile(outputPath + "valueEvaluation", sp);
		
		//visualize the value function and policy
		this.valueFunctionVisualize((QComputablePlanner)planner, p);
	
	}


	public void PolicyIteration(String outputPath){
		
		if(!outputPath.endsWith("/")){
			outputPath = outputPath + "/";
		}
	
                long startTime = System.nanoTime();	
                OOMDPPlanner planner = new PolicyIteration(domain, rf, tf, discount, hashingFactory, 
                                                           maxDelta, maxIterations, maxIterations);
		planner.planFromState(initialState);
	        long duration = System.nanoTime() - startTime;
                System.out.println("Running time:"+ (double)duration/1000000000+"s");
	
        	//create a Q-greedy policy from the planner
		Policy p = new GreedyQPolicy((QComputablePlanner)planner);
		
		//record the plan results to a file
		p.evaluateBehavior(initialState, rf, tf).writeToFile(outputPath + "policyEvaluation", sp);
		
		//visualize the value function and policy
		this.valueFunctionVisualize((QComputablePlanner)planner, p);
	
	}
	
	public void QLearning(String outputPath){
		
		if(!outputPath.endsWith("/")){
			outputPath = outputPath + "/";
		}

                long startTime = System.nanoTime();
                
                Policy learningPolicy = new EpsilonGreedy(epsilon);
		LearningAgent agent = new QLearning(domain, rf, tf, discount, hashingFactory, 
                                                    initialQ, learn_rate, learningPolicy, 1000);

                //learningPolicy = new EpsilonGreedy((QComputablePlanner)agent, epsilon);
                ((EpsilonGreedy)learningPolicy).setPlanner((OOMDPPlanner)agent);
                
                //Policy p = new GreedyQPolicy((QComputablePlanner)planner);

		//run learning for 100 episodes
		for(int i = 0; i < 100; i++){
			EpisodeAnalysis ea = agent.runLearningEpisodeFrom(initialState);
			ea.writeToFile(String.format("%se%03d", outputPath, i), sp); 
			System.out.println(i + ": " + ea.numTimeSteps());
		}
		long duration = System.nanoTime() - startTime;
                System.out.println("Running time:"+ (double)duration/1000000000+"s");

                learningPolicy.evaluateBehavior(initialState, rf, tf).writeToFile(outputPath 
                                                                         + "qLearn", sp);
		this.valueFunctionVisualize((QComputablePlanner)agent, learningPolicy);
	
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
		//spp.setRenderStyle(PolicyGlyphRenderStyle.DISTSCALED);
		spp.setRenderStyle(PolicyGlyphRenderStyle.MAXACTION);
		
		ValueFunctionVisualizerGUI gui = new ValueFunctionVisualizerGUI(allStates, svp, planner);
		gui.setSpp(spp);
		gui.setPolicy(p);
		gui.setBgColor(Color.GRAY);
		gui.initGUI();
	}


        public void experimenterAndPlotter(){
		
		LearningAgentFactory qLearningFactory = new LearningAgentFactory() {
			
			@Override
			public String getAgentName() {
				return "Q-learning";
			}
			
			@Override
			public LearningAgent generateAgent() {
                                return new QLearning(domain, rf, tf, discount, hashingFactory, 
                                                                         initialQ, learn_rate);
			}
		};


                LearningAgentFactory sarsaLearningFactory = new LearningAgentFactory() {
	
                	@Override
                	public String getAgentName() {
		                 return "SARSA";
                 	}
	
                 	@Override
                	public LearningAgent generateAgent() {
         		        return new SarsaLam(domain, rf, tf, discount, hashingFactory,
                                                                    initialQ, learn_rate, lambda);
	                }
                };                 


		StateGenerator sg = new ConstantStateGenerator(this.initialState);

		LearningAlgorithmExperimenter exp = new LearningAlgorithmExperimenter((SADomain)this.domain, 
			rf, sg, 10, 10, qLearningFactory, sarsaLearningFactory);

		exp.setUpPlottingConfiguration(500, 250, 2, 1000, 
			TrialMode.MOSTRECENTANDAVERAGE, 
			PerformanceMetric.CUMULATIVESTEPSPEREPISODE, 
			PerformanceMetric.AVERAGEEPISODEREWARD);

		exp.startExperiment();

		exp.writeStepAndEpisodeDataToCSV("expData");

	}     

}
