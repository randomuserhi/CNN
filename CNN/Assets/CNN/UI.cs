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
    public GameObject ResponseObj;
    public GameObject BackPropObj;

    private Slider PanSlider;
    private Slider RenderSlider;
    private Button UploadButton;
    private Toggle LearningToggle;
    private Toggle BackPropToggle;
    private InputField InputField;
    private InputField FilePathInputField;
    private InputField FileNameInputField;
    private Button ExportButton;
    private Button ImportButton;
    public static Text ResponseText;

    public void Start()
    {
        PanSlider = PanSliderObj.GetComponent<Slider>();
        RenderSlider = RenderSliderObj.GetComponent<Slider>();
        UploadButton = UploadButtonObj.GetComponent<Button>();
        LearningToggle = LearningToggleObj.GetComponent<Toggle>();
        BackPropToggle = BackPropObj.GetComponent<Toggle>();
        InputField = InputFieldObj.GetComponent<InputField>();
        ExportButton = ExportButtonObj.GetComponent<Button>();
        ImportButton = ImportButtonObj.GetComponent<Button>();
        FileNameInputField = FileNameInputFieldObj.GetComponent<InputField>();
        FilePathInputField = FilePathInputFieldObj.GetComponent<InputField>();
        ResponseText = ResponseObj.GetComponent<Text>();

        PanSlider.maxValue = CNN.N.Model.Count - 1;
        RenderSlider.wholeNumbers = true;
        RenderSlider.maxValue = 10;

        UploadButton.onClick.AddListener(delegate
        {
            CNN.Learning = false;
            LearningToggle.isOn = false;
            
            Matrix Answer = CNN.N.ForwardPropagate(InputField.text);
            CNN.N.Render(CNN.RenderLayer);
            UI.ResponseText.text = "Answer: " + CNN.TeacherBot.DataSet.IDToClass[Network.GetIndex(Answer.Buffer)];
        });
        LearningToggle.onValueChanged.AddListener(delegate
        {
            CNN.Learning = LearningToggle.isOn;
        });
        BackPropToggle.onValueChanged.AddListener(delegate
        {
            CNN.Test = BackPropToggle.isOn;
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
