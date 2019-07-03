using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.IO;
using ConvNet.Utilities;

namespace ConvNet.Parser
{
    public struct kvp
    {
        public string key;
        public string val;
        public bool used;

        public kvp(string key, string val)
        {
            this.key = key;
            this.val = val;
            this.used = false;
        }
    }

    // option_list.h
    public struct metadata
    {
        public int classes;
        public string[] names;
    }

    public class OptionList
    {

        public static LinkedList<kvp> read_data_cfg(string filename)
        {

            if (!File.Exists(filename)) Utils.file_error(filename);
            string line = "";
            int nu = 0;
            LinkedList<kvp> options = new LinkedList<kvp>();
            using (StreamReader streamReader = File.OpenText(filename))
            {
                while ((line = streamReader.ReadLine()) != null)
                {
                    ++nu;
                    line = Utils.RemoveWhitespace(line);
                    switch (line[0])
                    {
                        case '\0':
                        case '#':
                        case ';':
                            ;
                            break;
                        default:
                            if (!read_option(line, options))
                            {
                                Utils.Log(string.Format("Config file error line %d, could parse: %s\n", nu, line));
                            }
                            break;
                    }
                }
                streamReader.Close();
            }
            return options;
        }

        metadata get_metadata(string file)
        {
            metadata m = new metadata();
            LinkedList<kvp> options = read_data_cfg(file);

            string name_list = option_find_str(options, "names", null);
            if (name_list == null) name_list = option_find_str(options, "labels", null);
            if (name_list == null)
            {
                Utils.Log("No names or labels found\n");
            }
            else
            {
               // m.names = Data.get_labels(name_list);
            }
            m.classes = option_find_int(options, "classes", 2);

            Console.WriteLine(string.Format("Loaded - names_list: %s, classes = %d \n", name_list, m.classes));
            return m;
        }

        public static bool read_option(string s, LinkedList<kvp> options)
        {
            string[] optionString = s.Split('#');
            if (optionString.Length == 2)
            {
                option_insert(options, optionString[0], optionString[1]);
                return true;
            }
            else
                return false;

        }


        public static void option_insert(LinkedList<kvp> l, string key, string val)
        {
            kvp p = new kvp(key, val);
            l.Append(p);
        }

        public static void option_unused(LinkedList<kvp> l)
        {
            LinkedListNode<kvp> n = l.First;
            while (n != null)
            {
                kvp p = n.Value;
                if (!p.used)
                {
                    Utils.Log(string.Format("Unused field: '%s = %s'\n", p.key, p.val));
                }
                n = n.Next;
            }
        }

        public static string option_find(LinkedList<kvp> l, string key)
        {
            LinkedListNode<kvp> n = l.First;
            while (n != null)
            {
                kvp p = n.Value;
                if (p.key.Equals(key))
                {
                    p.used = true;
                    return p.val;
                }
                n = n.Next;
            }
            return null;
        }
        public static string option_find_str(LinkedList<kvp> l, string key, string def)
        {
            string v = option_find(l, key);
            if (v != null) return v;
            if (def != null) Utils.Log(string.Format("%s: Using default '%s'\n", key, def));
            return def;
        }

        public static string option_find_str_quiet(LinkedList<kvp> l, string key, string def)
        {
            string v = option_find(l, key);
            if (v != null) return v;
            return def;
        }

        public static int option_find_int(LinkedList<kvp> l, string key, int def)
        {
            string v = option_find(l, key);
            if (v != null) return int.Parse(v);
            Utils.Log(string.Format("%s: Using default '%d'\n", key, def));
            return def;
        }

        public static int option_find_int_quiet(LinkedList<kvp> l, string key, int def)
        {
            string v = option_find(l, key);
            if (v != null) return int.Parse(v);
            return def;
        }

        public static double option_find_float_quiet(LinkedList<kvp> l, string key, double def)
        {
            string v = option_find(l, key);
            if (v != null) return double.Parse(v);
            return def;
        }

        public static double option_find_float(LinkedList<kvp> l, string key, double def)
        {
            string v = option_find(l, key);
            if (v != null) return double.Parse(v);
            Utils.Log(string.Format("%s: Using default '%lf'\n", key, def));
            return def;
        }
    }
}
