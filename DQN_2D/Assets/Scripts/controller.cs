using UnityEngine;


public class controller : MonoBehaviour
{
    public float speed = 1;


    // Update is called once per frame
    void Update()
    {
        if (Input.GetKey(KeyCode.Z))
            transform.position = new Vector3(transform.position.x, transform.position.y, transform.position.z + Time.deltaTime * speed);
        if (Input.GetKey(KeyCode.S))
            transform.position = new Vector3(transform.position.x, transform.position.y, transform.position.z - Time.deltaTime * speed);
        if (Input.GetKey(KeyCode.Q))
            transform.position = new Vector3(transform.position.x - Time.deltaTime * speed, transform.position.y, transform.position.z);
        if (Input.GetKey(KeyCode.D))
            transform.position = new Vector3(transform.position.x + Time.deltaTime * speed, transform.position.y, transform.position.z);
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
