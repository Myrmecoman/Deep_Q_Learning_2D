using UnityEngine;


public class controller : MonoBehaviour
{
    // Update is called once per frame
    void Update()
    {
        if (Input.GetKeyDown(KeyCode.Z))
            transform.position = new Vector3(transform.position.x, transform.position.y, transform.position.z + 2);
        if (Input.GetKeyDown(KeyCode.S))
            transform.position = new Vector3(transform.position.x, transform.position.y, transform.position.z - 2);
        if (Input.GetKeyDown(KeyCode.Q))
            transform.position = new Vector3(transform.position.x - 2, transform.position.y, transform.position.z);
        if (Input.GetKeyDown(KeyCode.D))
            transform.position = new Vector3(transform.position.x + 2, transform.position.y, transform.position.z);

        transform.position = new Vector3(Mathf.Round(transform.position.x), transform.position.y, Mathf.Round(transform.position.z));
    }

    private void OnCollisionEnter(Collision col)
    {
        if (col.gameObject.CompareTag("GOOD"))
        {
            gameManager.instance.UpdateScore(1);
            Destroy(col.gameObject);
        }
        if (col.gameObject.CompareTag("BAD"))
        {
            gameManager.instance.UpdateScore(-1);
            Destroy(col.gameObject);
        }
    }
}
