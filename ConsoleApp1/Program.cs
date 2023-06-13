// See https://aka.ms/new-console-template for more information
using TrainDataGenerator;

Console.WriteLine("Hello, World!");
var alfabeto = new char[] { 'A', 'B' };
var largoInicial = 5;
var largoFinal = 10;

using (StreamWriter outfile = new StreamWriter("testdata.csv"))
{
    Combinations.GenerateCombinations(largoInicial, largoFinal, alfabeto,
        (x) => { outfile.WriteLine($"{(string)x};{LZ77Compression.CompressLZ77((string)x)}"); });
}