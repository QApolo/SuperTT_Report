public class UsuarioData {
    @SerializedName("id")
    private Integer id;
    @SerializedName("nombre")
    private String nombre;
    @SerializedName("apellido")
    private String apellidos;
    @SerializedName("username")
    private String email;
    @SerializedName("password")
    private String password;
    @SerializedName("responseCode")
    private Integer responseCode;
    @SerializedName("keyAuth")
    private String keyAuth;
    @SerializedName("currentPassword")
    private String currentPassword;

    public String getEmail() {
        return email;
    }
    public void setEmail(String email) {
        this.email = email;
    }
    public String getPassword() {
        return password;
    }
    public void setPassword(String password) {
        this.password = password;
    }
    public Integer getId() {
        return id;
    }
    public void setId(Integer id) {
        this.id = id;
    }
    public String getNombre() {
        return nombre;
    }
    public void setNombre(String nombre) {
        this.nombre = nombre;
    }
    public String getApellidos() {
        return apellidos;
    }
    public void setApellidos(String apellidos) {
        this.apellidos = apellidos;
    }
    public Integer getResponseCode() {
        return responseCode;
    }
    public void setResponseCode(Integer responseCode) {
        this.responseCode = responseCode;
    }
    public String getKeyAuth() {
        return keyAuth;
    }
    public void setKeyAuth(String keyAuth) {
        this.keyAuth = keyAuth;
    }
    public String getCurrentPassword() {
        return currentPassword;
    }
    public void setCurrentPassword(String currentPassword) {
        this.currentPassword = currentPassword;
    }
}