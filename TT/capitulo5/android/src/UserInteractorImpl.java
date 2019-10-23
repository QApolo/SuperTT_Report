public class UserInteractorImpl implements UserInteractor {
    public static final String TAG = UserInteractorImpl.class.getCanonicalName();
    private UserRepository repository;

    public UserInteractorImpl() {
        repository = new UserRepositoryImpl();
    }

    @Override
    public MutableLiveData<BusinessResult<UsuarioModel>> logIn(UsuarioModel usuarioModel) {
        BusinessResult<UsuarioModel> resultado = new BusinessResult<>();
        MutableLiveData<BusinessResult<UsuarioModel>> mutableLiveData = new MutableLiveData<>();
        usuarioModel.setValidPassword(RN002.isPasswordValid(usuarioModel.getPassword()));
        usuarioModel.setValidEmail(RN002.isEmailValid(usuarioModel.getEmail()));
        if (usuarioModel.getValidEmail() && usuarioModel.getValidPassword()) {
            UsuarioData usuarioData = new UsuarioData();
            usuarioData.setEmail(usuarioModel.getEmail());
            usuarioData.setPassword(usuarioModel.getPassword());
            mutableLiveData = repository.login(usuarioData);
        } else {
            resultado.setCode(ResultCodes.RN002);
            resultado.setResult(usuarioModel);
            mutableLiveData.setValue(resultado);
        }

        return mutableLiveData;
    }
}