using System;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Threading;


namespace ConvNet.Utilities
{
    class Utils
    {
        private const int EXIT_FAILURE = 0;
        private static ReaderWriterLockSlim _readWriteLock = new ReaderWriterLockSlim();

        public static string RemoveWhitespace(string input)
        {
            return new string(input.ToCharArray()
                .Where(c => !Char.IsWhiteSpace(c))
                .ToArray());
        }

        public static void Log(string logMessage)
        {
            using (StreamWriter w = File.AppendText("log.txt"))
            {
                w.Write("\r\nLog Entry : ");
                w.WriteLine($"{DateTime.Now.ToLongTimeString()} {DateTime.Now.ToLongDateString()}");
                w.WriteLine("  :");
                w.WriteLine($"  :{logMessage}");
                w.WriteLine("-------------------------------");
            }
        }

        public static void LogThreadSafe(string logMessage)
        {
            // Set Status to Locked
            _readWriteLock.EnterWriteLock();
            try
            {
                using (StreamWriter w = File.AppendText("log.txt"))
                {
                    w.Write("\r\nLog Entry : ");
                    w.WriteLine($"{DateTime.Now.ToLongTimeString()} {DateTime.Now.ToLongDateString()}");
                    w.WriteLine("  :");
                    w.WriteLine($"  :{logMessage}");
                    w.WriteLine("-------------------------------");
                    w.Close();
                }
            }
            finally
            {
                // Release lock
                _readWriteLock.ExitWriteLock();
            }
        }

        public static void file_error(string s)
        {
            Log(string.Format("Couldn't open file: %s\n", s));
            Environment.Exit(EXIT_FAILURE);
        }

        public static void error(string s)
        {
            Console.Error.WriteLine(s);
            Debug.Assert(false);
            Environment.Exit(EXIT_FAILURE);
        }

    }
}
