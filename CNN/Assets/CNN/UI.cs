using System.Collections;
using System.Collections.Generic;
using System.IO;
using UnityEngine;
using UnityEngine.UI;

public class UI : MonoBehaviour
{
    public GameObject CameraObj;
    public GameObject PanSliderObj;
    public GameObject RenderSliderObj;
    public GameObject UploadButtonObj;
    public GameObject LearningToggleObj;
    public GameObject InputFieldObj;
    public GameObject FilePathInputFieldObj;
    public GameObject FileNameInputFieldObj;
    public GameObject ExportButtonObj;
    public GameObject ImportButtonObj;

    private Slider PanSlider;
    private Slider RenderSlider;
    private Button UploadButton;
    private Toggle LearningToggle;
    private InputField InputField;
    private InputField FilePathInputField;
    private InputField FileNameInputField;
    private Button ExportButton;
    private Button ImportButton;

    public void Start()
    {
        PanSlider = PanSliderObj.GetComponent<Slider>();
        RenderSlider = RenderSliderObj.GetComponent<Slider>();
        UploadButton = UploadButtonObj.GetComponent<Button>();
        LearningToggle = LearningToggleObj.GetComponent<Toggle>();
        InputField = InputFieldObj.GetComponent<InputField>();
        ExportButton = ExportButtonObj.GetComponent<Button>();
        ImportButton = ImportButtonObj.GetComponent<Button>();
        FileNameInputField = FileNameInputFieldObj.GetComponent<InputField>();
        FilePathInputField = FilePathInputFieldObj.GetComponent<InputField>();

        PanSlider.maxValue = CNN.N.Model.Count - 1;
        RenderSlider.wholeNumbers = true;
        RenderSlider.maxValue = 10;

        UploadButton.onClick.AddListener(delegate
        {
            CNN.Learning = false;
            LearningToggle.isOn = false;
            
            CNN.N.ForwardPropagate(InputField.text);
            CNN.N.Render(CNN.RenderLayer);
        });
        LearningToggle.onValueChanged.AddListener(delegate
        {
            CNN.Learning = LearningToggle.isOn;
        });
        PanSlider.onValueChanged.AddListener(delegate
        {
            CameraObj.transform.position = new Vector3(PanSlider.value, 0, -10);
        });
        RenderSlider.onValueChanged.AddListener(delegate
        {
            CNN.RenderLayer = (int)RenderSlider.value;
        });
        ExportButton.onClick.AddListener(delegate
        {
            CNN.N.Export(FilePathInputField.text, FileNameInputField.text);
        });
        ImportButton.onClick.AddListener(delegate
        {
            CNN.N.Import(FilePathInputField.text + Path.DirectorySeparatorChar + FileNameInputField.text);
        });
    }
}
