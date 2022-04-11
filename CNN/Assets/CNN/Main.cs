using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

using UnityEngine;
using UnityEngine.UI;

public class Main : MonoBehaviour
{
    public GameObject Image;

    public void Start()
    {
        CNN.Start(Image.GetComponent<RawImage>());
    }

    public void FixedUpdate()
    {
        CNN.Update();
    }
}
