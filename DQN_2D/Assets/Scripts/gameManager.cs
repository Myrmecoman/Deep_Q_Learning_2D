using UnityEngine;
using UnityEngine.SceneManagement;
using UnityEngine.UI;


public class gameManager : MonoBehaviour
{
    public static gameManager instance;
    public Text score;
    public GameObject good;
    public GameObject bad;

    private int totalScore = 0;

    private void Awake()
    {
        // make this a singleton
        if (instance == null)
            instance = this;
        else
            Destroy(gameObject);
    }


    private void Start()
    {
        // spawn good and bad objects
        for (int i = -150; i < 150; i += 2)
        {
            for (int j = -150; j < 150; j += 2)
            {
                int rand = Random.Range(0, 2);
                if (rand == 0 && (Mathf.Abs(i) > 3 || Mathf.Abs(j) > 3))
                {
                    rand = Random.Range(0, 2);
                    if (rand == 0)
                    {
                        Instantiate(good, new Vector3(i, 0, j), Quaternion.identity);
                    }
                    else
                    {
                        Instantiate(bad, new Vector3(i, 0, j), Quaternion.identity);
                    }
                }
            }
        }
    }


    public void UpdateScore(int val)
    {
        totalScore += val;
        score.text = totalScore.ToString();
    }


    public void Reload()
    {
        SceneManager.LoadScene(0);
    }
}
