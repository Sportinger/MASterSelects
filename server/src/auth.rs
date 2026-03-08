use sha2::{Digest, Sha256};

pub struct AuthManager {
    token_hash: String,
}

impl AuthManager {
    pub fn new(token: &str) -> Self {
        Self {
            token_hash: hash_token(token),
        }
    }

    pub fn verify(&self, token: &str) -> bool {
        let provided_hash = hash_token(token);
        constant_time_eq(self.token_hash.as_bytes(), provided_hash.as_bytes())
    }
}

fn hash_token(token: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(token.as_bytes());
    hex::encode(hasher.finalize())
}

fn constant_time_eq(a: &[u8], b: &[u8]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    let mut diff = 0u8;
    for (x, y) in a.iter().zip(b.iter()) {
        diff |= x ^ y;
    }
    diff == 0
}
