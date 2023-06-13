

namespace TrainDataGenerator
{
    using System;
    using System.Collections.Generic;

    public class LZ77Compression
    {
        public static string CompressLZ77(string text)
        {
            string compressedText = "";

            int currentIndex = 0;
            while (currentIndex < text.Length)
            {
                // Find the longest matching substring
                int matchIndex = -1;
                int matchLength = 0;

                for (int i = currentIndex - 1; i >= 0; i--)
                {
                    int length = 0;

                    while (currentIndex + length < text.Length && text[i + length] == text[currentIndex + length])
                    {
                        length++;
                    }

                    if (length > matchLength)
                    {
                        matchIndex = i;
                        matchLength = length;
                    }
                }

                // Append the match information to the compressed text
                if (matchIndex != -1 && matchLength != 0)
                {
                    compressedText += $"({currentIndex - matchIndex - 1},{matchLength})";
                    currentIndex += matchLength;
                }
                else
                {
                    compressedText += text[currentIndex];
                    currentIndex++;
                }
            }

            return compressedText;
        }
    }
}
