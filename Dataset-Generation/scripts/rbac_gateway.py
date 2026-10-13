class RBACGateway:
    def __init__(self):
        self.roles = {
            "Simulation_Engineer": ["create", "verify", "encrypt"],
            "Privacy_Engineer": ["verify"],
            "ML_Researcher": ["read_processed"]
        }

    def verify_access(self, user_role, permission):
        if user_role in self.roles and permission in self.roles[user_role]:
            print(f"Access Granted for {user_role}")
            return True
        print(f"Access Denied: Role '{user_role}' cannot perform '{permission}'")
        return False
