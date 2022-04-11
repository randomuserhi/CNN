using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

using UnityEngine;

public class MemoryManager : MonoBehaviour
{
    public static List<Layer> DisposeList = new List<Layer>();

    private void OnApplicationQuit()
    {
        List<Layer> Disposing = new List<Layer>(DisposeList);
        for (int i = 0; i < Disposing.Count; i++)
        {
            Disposing[i].Dispose();
        }
        DisposeList.Clear();
    }
}