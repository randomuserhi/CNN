using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

using UnityEngine;

public class Main : MonoBehaviour
{
    public void Start()
    {
        CNN.Start();
    }

    public void FixedUpdate()
    {
        CNN.Update();
    }
}
