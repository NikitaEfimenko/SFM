import OpenGL.GL as gl
import OpenGL.GLU as glu
import OpenGL.GLUT as glut
import sys


class PCViewer:
    def __init__(self):
        self.ww = 800
        self.hh = 600
        self.xrot = 0
        self.yrot = 0
        self.mesh = False
        self.cameraCord = ([0, 0, 700], [0, 0, -1], [-1, 0, 0])

    def initPoints(self ,cor, clrCord, triag = None):
        self.cor = cor
        self.clrCord = clrCord
        self.triag = triag

    def meshPC(self):
        self.mesh = True

    def unmeshPC(self):
        self.mesh = False

    def __draw__(self):
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
        gl.glPushMatrix()

        gl.glRotatef(self.xrot, 1.0, 0.0, 0.0)
        gl.glRotatef(self.yrot, 0.0, 1.0, 0.0)
        gl.glEnableClientState(gl.GL_VERTEX_ARRAY)
        gl.glEnableClientState(gl.GL_COLOR_ARRAY)
        gl.glVertexPointer(3, gl.GL_FLOAT, 0, self.cor)
        gl.glColorPointer(3, gl.GL_FLOAT, 0, self.clrCord)
        gl.glDrawArrays(gl.GL_POINTS, 0, len(self.cor))

        if self.mesh:
            gl.glDrawElements(gl.GL_TRIANGLES, len(self.triag), gl.GL_UNSIGNED_BYTE, self.triag)

        gl.glDisableClientState(gl.GL_VERTEX_ARRAY)
        gl.glDisableClientState(gl.GL_COLOR_ARRAY)
        gl.glPopMatrix()
        glut.glutSwapBuffers()

    def __reshape__(self, w, h):
        gl.glViewport(0, 0, w, h)
        gl.glMatrixMode(gl.GL_PROJECTION)
        gl.glLoadIdentity()
        glu.gluPerspective(120, self.ww / self.hh, 0.3, 100000)
        gl.glMatrixMode(gl.GL_MODELVIEW)
        gl.glLoadIdentity()
        gl.glPointSize(2.0)
        x0, y0, z0 = self.cameraCord
        glu.gluLookAt(x0[0], x0[1], x0[2],
                      y0[0], y0[1], y0[2],
                      z0[0], z0[1], z0[2])

    def __specialkeys__(self, key, x, y):
        global xrot, yrot
        if key == glut.GLUT_KEY_UP:
            self.yrot -= 2
        if key == glut.GLUT_KEY_DOWN:
            self.yrot += 2
        if key == glut.GLUT_KEY_LEFT:
            self.xrot -= 2
        if key == glut.GLUT_KEY_RIGHT:
            self.xrot += 2
        if key == glut.GLUT_KEY_HOME:
            self.mesh = not self.mesh
        glut.glutPostRedisplay()


    def on(self):
        glut.glutInitDisplayMode(glut.GLUT_DOUBLE | glut.GLUT_RGB)
        glut.glutInitWindowSize(self.ww, self.hh)
        glut.glutInitWindowPosition(300, 100)
        glut.glutInit(sys.argv)
        glut.glutCreateWindow(b"PCViewer")
        glut.glutDisplayFunc(self.__draw__)
        gl.glEnable(gl.GL_DEPTH_TEST)
        glut.glutReshapeFunc(self.__reshape__)
        glut.glutSpecialFunc(self.__specialkeys__)
        glut.glutMainLoop()

