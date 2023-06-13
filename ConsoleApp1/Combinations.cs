using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace TrainDataGenerator
{
    public class Combinations
    {
        public static void GenerateCombinations(int minLength, int maxLength, char[] characters, Action<object> value)
        {

            GenerateCombinationsRecursive(minLength, maxLength, characters, new char[maxLength], 0, value);

        }

        private static void GenerateCombinationsRecursive(int minLength, int maxLength, char[] characters, char[] currentCombination, int currentIndex, Action<object> value)
        {
            if (currentIndex >= minLength)
            {
                value(new string(currentCombination, 0, currentIndex));
            }

            if (currentIndex < maxLength)
            {
                foreach (char c in characters)
                {
                    currentCombination[currentIndex] = c;
                    GenerateCombinationsRecursive(minLength, maxLength, characters, currentCombination, currentIndex + 1, value);
                }
            }
        }
    }
}
