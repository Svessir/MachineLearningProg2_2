/**
 * Created by Sverrir on 21.10.2016.
 */
public class Test {
    public static void main(String args[]) {
        RaceCar raceCar = new RaceCar();
        raceCar.qLearn();

        for(int i = 0; i < 10000; i++) {
            raceCar.drive();
        }
    }
}
