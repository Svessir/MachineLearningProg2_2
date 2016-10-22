import java.util.ArrayList;
import java.util.List;
import java.util.Random;

/**
 * Created by Sverrir on 20.10.2016.
 */
public class RaceCar {

    private RaceTrackSim raceTrackSimulator;

    private final int MIN_VX = -5;
    private final int MAX_VX = 5;
    private final int MIN_VY = -5;
    private final int MAX_VY = 5;

    private double epsilon = 1;
    private double learningRate = 0.0001;
    private double gamma = 1.0;

    private Random random;

    private final int MIN_ACTION = -1;
    private final int MAX_ACTION = 1;

    /**
     *  The complete set of actions with their corresponding weight vectors.
     */
    private double[][][] actionWeightPairs =
            {     /** Action    WeightVector*/
                    {{-1,-1},   {0,0,0,0,0,0,0,0,0}},
                    {{-1,0},    {0,0,0,0,0,0,0,0,0}},
                    {{-1,1},    {0,0,0,0,0,0,0,0,0}},
                    {{0,-1},    {0,0,0,0,0,0,0,0,0}},
                    {{0,0},     {0,0,0,0,0,0,0,0,0}},
                    {{0,1},     {0,0,0,0,0,0,0,0,0}},
                    {{1,-1},    {0,0,0,0,0,0,0,0,0}},
                    {{1,0},     {0,0,0,0,0,0,0,0,0}},
                    {{1,1},     {0,0,0,0,0,0,0,0,0}},
            };

    /**
     *  Constructor
     */
    public RaceCar() {
        raceTrackSimulator = new RaceTrackSim();
        random = new Random();
    }

    /**
     *
     */
    public void qLearn() {
        int i = 0;
        int j = 0;
        int numCrashes = 0;
        int crashOneInRow = 0;
        double[] state = null;
        while(true) {
            j++;
            long sleepTime = 1000;
            long sleepEnd;
            epsilon =  epsilon > 0 ? epsilon - 0.001 : 0;
           //if(i == 1)
           //    System.out.println("state: " + "(" + state[0] + ", " + state[1] + ", " + state[2] + ", " + state[3] + ")");
            state = raceTrackSimulator.startEpisode();
            //sleepEnd = System.currentTimeMillis() + sleepTime;
            //while(System.currentTimeMillis() < sleepEnd );
            System.out.println("\n" + j + " " + i);
            System.out.println("epsilon: " + epsilon);
            if(i != 0 && numCrashes == 0)
                crashOneInRow++;
            else
                crashOneInRow = 0;

            if(crashOneInRow == 1 && epsilon == 0)
                break;
            i = 0;
            numCrashes = 0;
            //printWeights();
            while(state[5] != 1.0D) {
                i++;
                double[] action = pickPolicyAction(state[0],state[1],state[2], state[3]);

                if(action == null)
                    break;

                double[] stateActionPair = new double[] {state[0], state[1], state[2], state[3], action[0], action[1]};
                double[] environmentResponse = raceTrackSimulator.simulate(stateActionPair);

                if(environmentResponse[4] == -5.0D) {
                    System.out.println("CRASH!");
                    numCrashes++;
                }
                else if(environmentResponse[5] == 1.0D && environmentResponse[4] == -1.0D)
                    System.out.println("FINISH!");

                updateQValue(stateActionPair, environmentResponse);
                state = environmentResponse;
            }
        }
    }

    /**
     * Updates the q value for a given state action pair.
     *
     * @param stateAction The state action pair who's q value is being updated.
     * @param environmentResponse The environment response for taking the action
     *                            in the state.
     */
    private void updateQValue(double[] stateAction, double[] environmentResponse) {
        double[] weights = getWeights(new double[] {stateAction[4], stateAction[5]});
        double maxQValueStatePrime = Double.NEGATIVE_INFINITY;
        double qValue = q(stateAction[0],stateAction[1],stateAction[2],stateAction[3], new double[] {stateAction[4],stateAction[5]});
        double targetValue;

        List<double[]> statePrimeActions = getActions(environmentResponse[0], environmentResponse[1],
                environmentResponse[2], environmentResponse[3]);

        if(environmentResponse[5] != 1.0D) {
            for(double[] action : statePrimeActions) {
                double qValuePrime = q(environmentResponse[0], environmentResponse[1],
                        environmentResponse[2], environmentResponse[3], action);
                if(qValuePrime > maxQValueStatePrime)
                    maxQValueStatePrime = qValuePrime;
            }
        }
        else
            maxQValueStatePrime = 0;

        //System.out.println();
        //System.out.println("state: " + "(" + stateAction[0] + ", " + stateAction[1] + ", " + stateAction[2] + ", " + stateAction[3] + ")");
        //System.out.println("action: " + "(" + stateAction[4] + "," + stateAction[5] + ")");
        //System.out.println("reward: " + environmentResponse[4]);
        //System.out.println("QValue :" + qValue);
        //System.out.println("QValuePrime :" + maxQValueStatePrime);
        //System.out.println("weights: (" + weights[0] + "," + weights[1] + "," + weights[2] + "," + weights[3] + "," + weights[4] + ")");

        targetValue = environmentResponse[4] + gamma * maxQValueStatePrime;
        double diff = targetValue - qValue;
        double[] featureVector = getFeatureVector(stateAction);

        for(int i = 0; i < weights.length; i++) {
            double delta = learningRate * diff * featureVector[i];
            weights[i] += delta;
            //System.out.println("weight[" + i + "] = " + weights[i] + " change: " + delta);
        }

        //System.out.println("weightsAfter: (" + weights[0] + "," + weights[1] + "," + weights[2] + "," + weights[3] + "," + weights[4] + ")");
        //System.out.println("QValueAfter: " + q(stateAction[0],stateAction[1],stateAction[2],stateAction[3], new double[] {stateAction[4],stateAction[5]}));
        //System.out.println();

        updateWeights(new double[] {stateAction[4], stateAction[5]}, weights);
    }

    /**
     * Gets all viable actions in a given state.
     *
     * @param x x coordinate of the car.
     * @param y y coordinate of the car.
     * @param v_x x velocity of the car.
     * @param v_y y velocity of the car.
     * @return List of all viable actions for the state.
     */
    private List<double[]> getActions(double x, double y, double v_x, double v_y) {
        List<double[]> viableActions = new ArrayList<double[]>();

        for(double[][] actionWeightPair : actionWeightPairs) {
            double[] action = actionWeightPair[0];
            double newVx = v_x + action[0];
            double newVy = v_y + action[1];
            if(newVx >= MIN_VX && newVx <= MAX_VX && newVy >= MIN_VY && newVy <= MAX_VY)
                viableActions.add(action);
        }

        return viableActions;
    }

    /**
     *
     * @param x The x coordinate of the car.
     * @param y The y coordinate of the car.
     * @param v_x The x velocity of the car.
     * @param v_y The y velocity of the car.
     * @param action The action being taken in the state.
     * @return The q value for the state action pair.
     */
    private double q(double x, double y, double v_x, double v_y, double[] action) {
        double[] weights = getWeights(action);
        return weights[0] + weights[1] * x + weights[2] * y + weights[3] * v_x + weights[4] * v_y;
    }

    /**
     * Picks an action according to epsilon greedy strategy/policy for a state.
     *
     * @param x The x coordinate of the car.
     * @param y The y coordinate of the car.
     * @param v_x The x velocity of the car.
     * @param v_y The y velocity of the car.
     * @return The action taken defined by the strategy/policy.
     */
    private double[] pickPolicyAction(double x, double y, double v_x, double v_y) {

        double epsilonGreedy = random.nextDouble();
        List<double[]> viableActions = getActions(x, y, v_x, v_y);

        if(viableActions.size() == 0)
            return null;

        double[] action = epsilonGreedy <= epsilon ? randomAction(viableActions) : greedyAction(x, y, v_x, v_y, viableActions);

        return action;
    }

    /**
     * Picks the greedy action from the list.
     * That is the action with the highest Q value for this state.
     *
     * @param x The x coordinate of the car.
     * @param y The y coordinate of the car.
     * @param v_x The x velocity of the car.
     * @param v_y The y velocity of the car.
     * @param viableActions The list of actions viable in this state.
     * @return The best action observed in this state up to date.
     */
    private double[] greedyAction(double x, double y, double v_x, double v_y, List<double[]> viableActions) {
        double[] maxAction = null;
        double maxQValue = Double.NEGATIVE_INFINITY;

        for(double[] action : viableActions) {
            double qValue = q(x,y,v_x,v_y,action);

            if(qValue >= maxQValue) {
                maxQValue = qValue;
                maxAction = action;
            }
        }

        return maxAction;
    }

    /**
     * Picks a random action from the actions list provided.
     *
     * @param viableActions The list of actions the random sample will be picked from.
     * @return The random action from the list.
     */
    private double[] randomAction(List<double[]> viableActions) {
        int randomIndex = random.nextInt(viableActions.size());
        return viableActions.get(randomIndex);
    }

    /**
     * Gets the weight vector for the action.
     *
     * @param action An action in the complete set of actions.
     * @return The weight vector for the action.
     */
    private double[] getWeights(double[] action) {
        int index = getActionIndex(action);
        return actionWeightPairs[index][1];
    }

    /**
     * Updates the weights vector for a action.
     *
     * @param action The action of the weight vector.
     * @param weights The new weights.
     */
    private void updateWeights(double[] action, double[] weights) {
        int index = getActionIndex(action);
        actionWeightPairs[index][1] = weights;
    }

    /**
     * Gets the index of this action in the
     * array of all actions.
     *
     * @param action The action being queried for index.
     * @return The index of the action in the array of all actions.
     * @throws RuntimeException If the action is not within the array.
     */
    private int getActionIndex(double[] action) throws RuntimeException {
        if(action.length != 2 || action[0] < MIN_ACTION || action[0] > MAX_ACTION || action[1] < MIN_ACTION || action[1] > MAX_ACTION)
            throw new RuntimeException("Querying for an illegal action.");

        return (MAX_ACTION - MIN_ACTION + 1) * ((int)action[0] - MIN_ACTION) + ((int)action[1] - MIN_ACTION);
    }

    /**
     * Gets the feature vector for a state.
     *
     * @param state The state being transformed to a feature vector.
     * @return The feature vector of the state.
     */
    private double[] getFeatureVector(double[] state) {
        return new double[] {1.0D, state[0], state[1], state[2], state[3], Math.sqrt(Math.pow(state[2], 2) + Math.pow(state[3], 2)), state[0] * state[1],
        state[0] * state[0], state[1] * state[1]};
    }

    private void printWeights() {
        //System.out.println();
        for(double[][] pair : actionWeightPairs) {
            double[] weights = pair[1];
            System.out.print("( ");
            for(double weight : weights)
                System.out.print( weight + ", ");
            System.out.println(" )");
        }
        System.out.println();
    }
}
