class SessionGuardService {
  SessionGuardService._();

  static final SessionGuardService instance = SessionGuardService._();

  int _externalOperationDepth = 0;

  bool get isInExternalOperation => _externalOperationDepth > 0;

  void beginExternalOperation() {
    _externalOperationDepth++;
  }

  void endExternalOperation() {
    if (_externalOperationDepth <= 0) {
      _externalOperationDepth = 0;
      return;
    }
    _externalOperationDepth--;
  }
}
