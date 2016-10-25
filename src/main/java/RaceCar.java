import java.util.ArrayList;
import java.util.List;
import java.util.Random;

/**
 * Created by Sverrir on 20.10.2016.
 */
public class RaceCar {

    private RaceTrackSim raceTrackSimulator;

    private final int MIN_VX = -4;
    private final int MAX_VX = 4;
    private final int MIN_VY = -4;
    private final int MAX_VY = 4;

    private double epsilon = 0.2;
    private double learningRate = 0.00001; //0.00001;
    private double gamma = 1.0;

    private Random random;

    private final int MIN_ACTION = -1;
    private final int MAX_ACTION = 1;

    private int maxX = 100;
    private int maxY = 100;

    /**
     *  The complete set of actions with their corresponding weight vectors.
     */
    private double[][][] actionWeightPairs =
            {     /** Action    WeightVector*/
                    {{-1,-1},   {0,0,0,0,0,0}},
                    {{-1,0},    {0,0,0,0,0,0}},
                    {{-1,1},    {0,0,0,0,0,0}},
                    {{0,-1},    {0,0,0,0,0,0}},
                    {{0,0},     {0,0,0,0,0,0}},
                    {{0,1},     {0,0,0,0,0,0}},
                    {{1,-1},    {0,0,0,0,0,0}},
                    {{1,0},     {0,0,0,0,0,0}},
                    {{1,1},     {0,0,0,0,0,0}},
            };

    /** Tiling stuff*/
    private final double minVelocity = -6;
    private double[][] xTilesWeights;
    private double[][] yTilesWeights;
    private double[][] vxTilesWeights;
    private double[][] vyTilesWeights;
    private double[][] speedTilesWeights;

    private int j = 0;
    /**
     *  Constructor
     */
    public RaceCar() {
        raceTrackSimulator = new RaceTrackSim();
        random = new Random();

        // Create weights for every tile
        int maxSpeedLength = (int)Math.ceil(Math.sqrt(Math.max(Math.pow(MIN_VX, 2), Math.pow(MAX_VX, 2))
                + Math.max(Math.pow(MIN_VY, 2), Math.pow(MAX_VY, 2))));

        xTilesWeights = new double[100][9];
        yTilesWeights = new double[100][9];
        vxTilesWeights = new double[13][9];
        vyTilesWeights = new double[13][9];
        speedTilesWeights = new double[maxSpeedLength][9];

        j = 0;
    }

    /**
     * Initiate q learning.
     */
    public void qLearn() {
        double[] state = null;
        int i = 0;
        int crashThreshold = 5;
        int numberOfEpisodesLowerThanCrashThreshold = 0;
        int numberOfCrashes = Integer.MAX_VALUE;
        int maxNumberOfCrashes=0;
        while(numberOfEpisodesLowerThanCrashThreshold < 100 && i < 10000000) {
            numberOfEpisodesLowerThanCrashThreshold = numberOfCrashes < crashThreshold ? numberOfEpisodesLowerThanCrashThreshold + 1: 0;
            i++;
            maxNumberOfCrashes = numberOfCrashes > maxNumberOfCrashes ? numberOfCrashes : maxNumberOfCrashes;
            if(i % 10000 == 0) {
                System.out.println(i + " numberOfEpisodesLowerThanCrashThreshold : " + numberOfEpisodesLowerThanCrashThreshold
                                    + " numberOfCrashes: " + maxNumberOfCrashes);
                maxNumberOfCrashes = 0;
            }
            numberOfCrashes = 0;

            state = raceTrackSimulator.startEpisode();
            while(state[5] != 1.0D) {

                double[] action = pickPolicyAction(state[0],state[1],state[2], state[3]);

                double[] stateActionPair = new double[] {state[0], state[1], state[2], state[3], action[0], action[1]};
                double[] environmentResponse = raceTrackSimulator.simulate(stateActionPair);

                //if(environmentResponse[4] == -5.0D)
                //    System.out.println("CRASH");
                //if(environmentResponse[5] == 1.0D)
                //    System.out.println("FINISH");

                numberOfCrashes += environmentResponse[4] == -5.0D ? 1 : 0;

                updateQValue(stateActionPair, environmentResponse);
                state = environmentResponse;
            }
        }
    }

    public void drive() {
        double[] state = raceTrackSimulator.startEpisode();
        double oldEpsilon = epsilon;
        epsilon = 0;
        do {
            double[] action = pickPolicyAction(state[0],state[1],state[2], state[3]);

            double[] stateActionPair = new double[] {state[0], state[1], state[2], state[3], action[0], action[1]};
            double[] environmentResponse = raceTrackSimulator.simulate(stateActionPair);

            if(environmentResponse[4] == -5.0D)
                System.out.println("CRASH");

            state = environmentResponse;
        } while(state[5] != 1.0D);
        System.out.println("FINISH\n");
        epsilon = oldEpsilon;
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

        targetValue = environmentResponse[4] + gamma * maxQValueStatePrime;
        double error = targetValue - qValue;
        double[] featureVector = getFeatureVector(stateAction);

        for(int i = 0; i < weights.length; i++) {
            double delta = learningRate * error * featureVector[i];
            weights[i] += delta;
        }

        updateWeights(new double[] {stateAction[4], stateAction[5]}, weights);

        //targetValue = environmentResponse[4] + gamma * maxQValueStatePrime;
        //double error = targetValue - qValue;
        //updateTileWeights(stateAction[0],stateAction[1],stateAction[2],stateAction[3], new double[] {stateAction[4],stateAction[5]}, error);
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
     * Gets the Q value of a state action pair.
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
        double[] features = getFeatureVector(new double[] {x,y,v_x,v_y});
        return weights[0] * features[0] + weights[1] * features[1] +
                weights[2] * features[2] + weights[3] * features[3] + weights[4] * features[4];
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
        return new double[] {1.0D, state[0], state[1], state[2], state[3],
                Math.sqrt(Math.pow(state[2], 2) + Math.pow(state[3], 2))};
    }

    /**
     * Prints all weights of actions.
     */
    private void printWeights() {
        for(double[][] pair : actionWeightPairs) {
            double[] weights = pair[1];
            System.out.print("( ");
            for(double weight : weights)
                System.out.print( weight + ", ");
            System.out.println(" )");
        }
        System.out.println();
    }


    /**
     * Gets the Q vlaue of an state action pair
     * via tiling method.
     *
     * @param x The x coordinate of the car.
     * @param y The y coordinate of the car.
     * @param v_x The x velocity of the car.
     * @param v_y The y velocity of the car.
     * @param action The action being taken in the state.
     * @return The q value for the state action pair.
     */
    private double getQValue(double x, double y, double v_x, double v_y, double[] action) {
        int actionIndex = getActionIndex(action);
        int speedIndex = (int)Math.floor(Math.sqrt(Math.pow(v_x, 2) + Math.pow(v_y, 2)));
        return xTilesWeights[(int)Math.floor(x)][actionIndex] + yTilesWeights[(int)Math.floor(y)][actionIndex]
                + vxTilesWeights[(int)Math.floor(v_x - minVelocity)][actionIndex]
                + vyTilesWeights[(int)Math.floor(v_y - minVelocity)][actionIndex]
                + speedTilesWeights[speedIndex][actionIndex];
    }


    /**
     * Updates the weights of tiles
     * according to error.
     *
     * @param x The x coordinate of the car.
     * @param y The y coordinate of the car.
     * @param v_x The x velocity of the car.
     * @param v_y The y velocity of the car.
     * @param action The action being taken in the state.
     * @param error The error between current q value
     *              and observed q value.
     */
    private void updateTileWeights(double x, double y, double v_x, double v_y, double[] action, double error) {
        int actionIndex = getActionIndex(action);
        int speedIndex = (int)Math.floor(Math.sqrt(Math.pow(v_x, 2) + Math.pow(v_y, 2)));
        double dividedLearningRate = learningRate/5;

        xTilesWeights[(int)Math.floor(x)][actionIndex] += dividedLearningRate * error;
        yTilesWeights[(int)Math.floor(y)][actionIndex] += dividedLearningRate * error;
        vxTilesWeights[(int)Math.floor(v_x - minVelocity)][actionIndex] += dividedLearningRate * error;
        vyTilesWeights[(int)Math.floor(v_y - minVelocity)][actionIndex] += dividedLearningRate * error;
        speedTilesWeights[speedIndex][actionIndex] += dividedLearningRate * error;
    }

    /**
     * Prints tile weights.
     */
    private void printTileWeights() {
        System.out.print("{ ");
        for(double weights[] : xTilesWeights) {
            System.out.print(weights[0] + " ");
        }
        System.out.println("}");

        System.out.print("{ ");
        for(double weights[] : yTilesWeights) {
            System.out.print(weights[0] + " ");
        }
        System.out.println("}");

        System.out.print("{ ");
        for(double weights[] : vxTilesWeights) {
            System.out.print(weights[0] + " ");
        }
        System.out.println("}");

        System.out.print("{ ");
        for(double weights[] : vyTilesWeights) {
            System.out.print(weights[0] + " ");
        }
        System.out.println("}");

        System.out.print("{ ");
        for(double weights[] : speedTilesWeights) {
            System.out.print(weights[0] + " ");
        }
        System.out.println("}");
        System.out.println();
    }
}
