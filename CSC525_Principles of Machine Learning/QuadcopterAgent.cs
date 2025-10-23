using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Sensors;

/// <summary>
/// A Quadcopter Machine Learning Agent
/// </summary>
public class QuadcopterAgent : Agent
{
    [Tooltip("Force to apply for movement")]
    public float moveForce = 20f;

    [Tooltip("Torque to apply for rotation")]
    public float rotateTorque = 10f;

    [Tooltip("Transform at the center of the quadcopter")]
    public Transform quadcopterCenter;

    [Tooltip("Whether this is training mode or gameplay mode")]
    public bool trainingMode;

    [Tooltip("Target object to fly to")]
    public GameObject target;

    [Tooltip("Whether to end episode on collision")]
    public bool endEpisodeOnCollision = true;
    
    [Tooltip("Whether to end episode on premature landing")]
    public bool endEpisodeOnLand = true;

    [Header("Spawn Settings")]
    [Tooltip("Center point of map (reference point for spawning)")]
    public GameObject center;

    [Tooltip("Fixed spawn height of drone")]
    public float DroneSpawnHeight = 10f;

    [Tooltip("Fixed spawn height of drone")]
    public float TargetSpawnHeight = 1f;

    private Vector3 line1Start => center.transform.position + new Vector3(-30, 0, 42);
    private Vector3 line1End => center.transform.position + new Vector3(40, 0, 42);

    private Vector3 line2Start => center.transform.position + new Vector3(0, 0, 42);
    private Vector3 line2End => center.transform.position + new Vector3(0, 0, -42);

    private Vector3 line3Start => center.transform.position + new Vector3(-30, 0, -42);
    private Vector3 line3End => center.transform.position + new Vector3(30, 0, -42);


    // The rigidbody of the agent
    private Rigidbody rBody;

    // Whether the agent is frozen (intentionally not flying)
    private bool frozen = false;

    // The current target position
    private Vector3 targetPosition;

    // Distance to target when episode started
    private float initialDistanceToTarget;

    // Add these new fields
    private float initialGroundDistance;  // Ground distance (XZ only) at episode start
    private float lastGroundDistance;     // Ground distance from previous step
    private float last3DDistance;         // 3D distance from previous step

    /// <summary>
    /// Initialize the agent
    /// </summary>
    public override void Initialize()
    {
        rBody = GetComponent<Rigidbody>();

        // If not training mode, no max step, play forever
        if (!trainingMode) MaxStep = 0;
    }

    /// <summary>
    /// Calculate ground distance (XZ plane only) between two points
    /// </summary>
    private float CalculateGroundDistance(Vector3 from, Vector3 to)
    {
        Vector2 fromFlat = new Vector2(from.x, from.z);
        Vector2 toFlat = new Vector2(to.x, to.z);
        return Vector2.Distance(fromFlat, toFlat);
    }

    /// <summary>
    /// Reset the agent when an episode begins
    /// </summary>
    public override void OnEpisodeBegin()
    {
        if (frozen) return;

        // Reset velocities
        rBody.linearVelocity = Vector3.zero;
        rBody.angularVelocity = Vector3.zero;

        // Reset rotation
        transform.rotation = Quaternion.Euler(0f, Random.Range(0f, 360f), 0f);

        // Move agent to random position
        MoveToRandomPosition();

        // Move target to random position
        MoveTargetToRandomPosition();

        // Store initial distances
        initialDistanceToTarget = Vector3.Distance(quadcopterCenter.position, targetPosition);
        initialGroundDistance = CalculateGroundDistance(quadcopterCenter.position, targetPosition);
        
        // Initialize last distances
        last3DDistance = initialDistanceToTarget;
        lastGroundDistance = initialGroundDistance;
    }

    /// <summary>
    /// Called when an action is received from either the player input or the neural network
    /// 
    /// vectorAction[i] represents:
    /// Index 0: move vector x (+1 = right, -1 = left)
    /// Index 1: move vector y (+1 = up, -1 = down)
    /// Index 2: move vector z (+1 = forward, -1 = backward)
    /// Index 3: rotation y (+1 = rotate right, -1 = rotate left)
    /// </summary>
    public override void OnActionReceived(ActionBuffers actions)
    {
        // Don't take actions if frozen
        if (frozen) return;

        // Convert actions to movement and rotation
        Vector3 move = new Vector3(
            actions.ContinuousActions[0],
            actions.ContinuousActions[1],
            actions.ContinuousActions[2]
        );

        // Add movement force
        rBody.AddForce(move * moveForce);

        // Add rotation torque
        float rotateAmount = actions.ContinuousActions[3];
        rBody.AddTorque(transform.up * rotateAmount * rotateTorque);

        // Calculate current distances
        float current3DDistance = Vector3.Distance(quadcopterCenter.position, targetPosition);
        float currentGroundDistance = CalculateGroundDistance(quadcopterCenter.position, targetPosition);

        // Reached target (using both ground and height thresholds)
        if (currentGroundDistance < 2f && Mathf.Abs(quadcopterCenter.position.y - targetPosition.y) < 2f)
        {
            AddReward(2.0f);
            EndEpisode();
            return;
        }

        // lands prematurely
        if (transform.position.y < .9f)
        {
            AddReward(-1.0f);
            
            if(endEpisodeOnLand)
            {
                EndEpisode();
            }
            
            return;
        }

        // Calculate distance changes
        float delta3DDistance = last3DDistance - current3DDistance;
        float deltaGroundDistance = lastGroundDistance - currentGroundDistance;

        // Reward for decreasing 3D distance to target
        if (delta3DDistance > 0)
        {
            AddReward(2.0f * delta3DDistance / initialDistanceToTarget);
        }

        // Penalty for increasing ground distance
        if (deltaGroundDistance < 0)
        {
            AddReward(-0.5f * Mathf.Abs(deltaGroundDistance) / initialGroundDistance);
        }

        if (current3DDistance < 5f)
        {
            float closeReward=0.2f*Mathf.Exp(-current3DDistance);
            AddReward(closeReward);
        }
    
        // Small negative reward each step to encourage reaching target faster
        AddReward(-0.001f);

        // Update last distances for next step
        last3DDistance = current3DDistance;
        lastGroundDistance = currentGroundDistance;
    }

    /// <summary>
    /// Collect vector observations from the environment
    /// </summary>
    public override void CollectObservations(VectorSensor sensor)
    {
        // Observe quadcopter position (3 observations)
        sensor.AddObservation(transform.localPosition);

        // Observe quadcopter rotation (4 observations)
        sensor.AddObservation(transform.localRotation.normalized);

        // Observe quadcopter velocity (3 observations)
        sensor.AddObservation(rBody.linearVelocity);

        // Observe quadcopter angular velocity (3 observations)
        sensor.AddObservation(rBody.angularVelocity);

        // Direction to target (3 observations)
        Vector3 toTarget = (targetPosition - quadcopterCenter.position).normalized;
        sensor.AddObservation(toTarget);

        // 3D distance to target (1 observation)
        float distanceToTarget = Vector3.Distance(quadcopterCenter.position, targetPosition);
        sensor.AddObservation(distanceToTarget);

        // Ground distance to target (1 observation)
        float groundDistance = CalculateGroundDistance(quadcopterCenter.position, targetPosition);
        sensor.AddObservation(groundDistance);

        // Height difference to target (1 observation)
        float heightDifference = targetPosition.y - quadcopterCenter.position.y;
        sensor.AddObservation(heightDifference);

        // Total observations: 19
    }

    /// <summary>
    /// When Behavior Type is set to "Heuristic Only" on the agent's Behavior Parameters,
    /// this function will be called. Its return values will be fed into
    /// <see cref="OnActionReceived"/>
    /// </summary>
    public override void Heuristic(in ActionBuffers actionsOut)
    {
        // Create placeholders for all movement/turning
        Vector3 forward = Vector3.zero;
        Vector3 left = Vector3.zero;
        Vector3 up = Vector3.zero;
        float yaw = 0f;

        // Convert keyboard inputs to movement and turning
        // All values should be between -1 and +1

        // Forward/backward
        if (Input.GetKey(KeyCode.W)) forward = transform.forward;
        else if (Input.GetKey(KeyCode.S)) forward = -transform.forward;

        // Left/right
        if (Input.GetKey(KeyCode.A)) left = -transform.right;
        else if (Input.GetKey(KeyCode.D)) left = transform.right;

        // Up/down
        if (Input.GetKey(KeyCode.UpArrow)) up = transform.up;
        else if (Input.GetKey(KeyCode.DownArrow)) up = -transform.up;

        // Turn left/right
        if (Input.GetKey(KeyCode.LeftArrow)) yaw = -1f;
        else if (Input.GetKey(KeyCode.RightArrow)) yaw = 1f;

        // Combine the movement vectors and normalize
        Vector3 combined = (forward + left + up).normalized;

        // Add the 3 movement values, pitch, and yaw to the actionsOut array
        actionsOut.ContinuousActions.Array[0] = combined.x;
        actionsOut.ContinuousActions.Array[1] = combined.y;
        actionsOut.ContinuousActions.Array[2] = combined.z;
        actionsOut.ContinuousActions.Array[3] = yaw;
    }

    /// <summary>
    /// Move the agent to a random position
    /// </summary>
    private void MoveToRandomPosition()
    {
        // Pick a random line (0, 1, or 2)
        int lineChoice = Random.Range(0, 3);
        Vector3 start, end;

        switch (lineChoice)
        {
            case 0:
                start = line1Start;
                end = line1End;
                break;
            case 1:
                start = line2Start;
                end = line2End;
                break;
            default:
                start = line3Start;
                end = line3End;
                break;
        }

        // Get random point along the chosen line
        float t = Random.value; // Random value between 0 and 1
        Vector3 position = Vector3.Lerp(start, end, t);
        
        // Set the fixed height
        position.y = DroneSpawnHeight;

        transform.position = position;
    }

    /// <summary>
    /// Move the target to a random position
    /// </summary>
    private void MoveTargetToRandomPosition()
    {
        // Pick a random line (0, 1, or 2)
        int lineChoice = Random.Range(0, 3);
        Vector3 start, end;

        switch (lineChoice)
        {
            case 0:
                start = line1Start;
                end = line1End;
                break;
            case 1:
                start = line2Start;
                end = line2End;
                break;
            default:
                start = line3Start;
                end = line3End;
                break;
        }

        // Get random point along the chosen line
        float t = Random.value;
        targetPosition = Vector3.Lerp(start, end, t);
        
        // Set the fixed height
        targetPosition.y = TargetSpawnHeight;

        if (target != null)
        {
            target.transform.position = targetPosition;
        }
    }


    /// <summary>
    /// Called when the drone collides with another object
    /// </summary>
    private void OnCollisionEnter(Collision collision)
    {
        if (frozen) return;

        // Don't penalize collisions with the target
        if (collision.gameObject == target) return;

        // Apply penalty
        AddReward(-1.0f);

        // Optionally end the episode
        if (endEpisodeOnCollision)
        {
            EndEpisode();
        }
    }


    /// <summary>
    /// Prevent the agent from moving and taking actions
    /// </summary>
    public void FreezeAgent()
    {
        Debug.Assert(trainingMode == false, "Freeze/Unfreeze not supported in training");
        frozen = true;
        rBody.Sleep();
    }

    /// <summary>
    /// Resume agent movement and actions
    /// </summary>
    public void UnfreezeAgent()
    {
        Debug.Assert(trainingMode == false, "Freeze/Unfreeze not supported in training");
        frozen = false;
        rBody.WakeUp();
    }
}